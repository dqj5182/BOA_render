import os
import cv2
import copy
import time
import torch
import random
import joblib
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
from tqdm import tqdm
import learn2learn as l2l

# import torchgeometry as tgm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import config
import constants
from models.hmr import hmr
from datasets.pw3d import PW3D
from datasets.h36 import H36M
from datasets.mpi_3dhp import HP3D
from utils.smpl import SMPL, get_smpl_faces
from utils.pose_utils import reconstruction_error
from utils.geometry import perspective_projection, rotation_matrix_to_angle_axis, estimate_translation, batch_rodrigues
from utils.losses import *
from utils.render_demo import Renderer

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, default='experiments', help='common dir of each experiment')
parser.add_argument('--name', type=str, default='', help='exp name')
parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument('--model_file', type=str, default='data/basemodel.pt', help='base model')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--dataset_name', type=str, default='3dpw', choices=['3dpw', 'mpi-inf-3dhp'], help='test set name')
parser.add_argument('--img_res', type=int, default=224, help='image resolution')
parser.add_argument('--saveimg', action='store_true', default=False, help='save visilized results? default: False')

# # baseline hyper-parameters
parser.add_argument('--lr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2')
parser.add_argument('--s2dloss_weight', type=float, default=10, help='weight of reprojection kp2d loss')
parser.add_argument('--shapeloss_weight', type=float, default=2e-6, help='weight of shape prior')
parser.add_argument('--gmmpriorloss_weight', type=float, default=1e-4, help='weight of pose prior(GMM)')
parser.add_argument('--labelloss_weight', type=float, default=0.1, help='weight of h36m loss')

# mean-teacher hyper-parameters
parser.add_argument('--ema_decay', type=float, default=0.1, help='ema_decay * T + (1-ema_decay) * M')
parser.add_argument('--consistentloss_weight', type=float, default=0.1, help='weight of consistent loss')
parser.add_argument('--consistent_s3d_weight', type=float, default=5, help='weight of shape prior')
parser.add_argument('--consistent_s2d_weight', type=float, default=5, help='weight of consistent loss')
parser.add_argument('--consistent_pose_weight', type=float, default=1, help='weight of pose prior(GMM)')
parser.add_argument('--consistent_beta_weight', type=float, default=0.001, help='weight of h36m loss')

# bilevel
parser.add_argument('--metalr', type=float, default=8e-6, help='learning rate')   # lower learning rate
parser.add_argument('--prev_n', type=int, default=5)
parser.add_argument('--motionloss_weight', type=float, default=0.1)

# predefined variables
device = torch.device('cuda')
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
smpl_neutral = SMPL(config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', batch_size=1, create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', batch_size=1, create_transl=False).to(device)
# -- end

# mean teacher help functions
def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

def seed_everything(seed=42):
    """ we need set seed to ensure that all model has same initialization
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    print('---> seed has been set')

def create_model(ema=False):
    model = hmr(config.SMPL_MEAN_PARAMS)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def save_model(model, optimizer, name):
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, name)
    print(f'checkpoint file: {name} is saved')

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
# -- end

class Adaptator():
    def __init__(self, options):
        self.options = options
        self.exppath = osp.join(self.options.expdir, self.options.name)
        self.summary_writer = SummaryWriter(self.exppath)
        self.device = torch.device('cuda')
        seed_everything(self.options.seed)

        model = create_model()
        self.model = l2l.algorithms.MAML(model, lr=self.options.metalr, first_order=False).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.lr, betas=(self.options.beta1, self.options.beta2))
        # use meanteacher
        self.ema_model = create_model(ema=True).to(self.device)

        # load model
        checkpoint = torch.load(self.options.model_file)
        self.model.load_state_dict(checkpoint['model'], strict=True)
        # mean-teacher
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        self.ema_model.load_state_dict(checkpoint['model'], strict=True)
        print('pretrained CKPT has been load')

        # build dataloders
        if '3dpw' in self.options.dataset_name:
            # 3dpw
            self.pw3d_dataset = PW3D(self.options, '3dpw')
            self.pw3d_dataloader = DataLoader(self.pw3d_dataset, batch_size=self.options.batch_size, shuffle=False, num_workers=0)
            #self.pw3d_dataloader = DataLoader(self.pw3d_dataset, batch_size=self.options.batch_size, shuffle=False, num_workers=16)
        elif 'mpi-inf' in self.options.dataset_name:
            # 3DHP
            self.pw3d_dataset = HP3D(self.options, 'mpi-inf-3dhp')
            self.pw3d_dataloader = DataLoader(self.pw3d_dataset, batch_size=self.options.batch_size, shuffle=False, num_workers=8)
        # h36m
        self.h36m_dataset = H36M(self.options, 'h36m')
        self.h36m_dataloader = DataLoader(self.h36m_dataset, batch_size=self.options.batch_size, shuffle=False, num_workers=8)
        print('dataset has been created')

    def reloadmodel(self,):
        checkpoint = torch.load(self.options.model_file)
        checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()), lr=self.options.lr, betas=(self.options.beta1, self.options.beta2))
        return

    def set_dropout_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            m.eval()

    def freeze_dropout(self,):
        self.model.apply(self.set_dropout_eval)
        self.ema_model.apply(self.set_dropout_eval)
    
    def freeze_model_except_groupnorm(self):
        for name, param in self.model.named_parameters():
            if 'bn' not in name:
                param.requires_grad = False
        self.freeze_gn()

    def write_summaries(self, datas, stepcount, is_test=False):
        for k, v in datas.items():
            if is_test:
                k = 'test/'+k 
            self.summary_writer.add_scalar(k, v, stepcount)

    def inference(self,):
        # we split the inference stage to adaption stage and test stage.
        joint_mapper_h36m = constants.H36M_TO_J17 if self.options.dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if self.options.dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

        h36m_loader = iter(self.h36m_dataloader)

        mpjpe, pampjpe = [], []
        uposed_mesh_error, posed_mesh_error = [],[]
        self.history_info = {}
        for step, pw3d_batch in tqdm(enumerate(self.pw3d_dataloader), total=len(self.pw3d_dataloader)):
            self.global_step = step
            
            # load h36m data
            try:
                h36m_batch = next(h36m_loader)
            except StopIteration:
                h36m_loader = iter(self.h36m_dataloader)
                h36m_batch = next(h36m_loader)
            h36m_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in h36m_batch.items()}

            pw3d_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in pw3d_batch.items()}

            # freeze dropout
            self.model.train()
            self.ema_model.train()
            self.freeze_dropout()
            
            if self.global_step == 0:
                print('Model parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad) /1000/1000)

            # adaptation
            self.optimizer.zero_grad()
            upperlevel_loss= self.meta_adaptation(pw3d_batch, h36m_batch)
            upperlevel_loss.backward()
            self.optimizer.step()
            update_ema_variables(self.model, self.ema_model, self.options.ema_decay, step)

            # test using the adapted model
            eval_res = self.test(pw3d_batch, joint_mapper_gt, joint_mapper_h36m)
            mpjpe.append(eval_res['mpjpe'])
            pampjpe.append(eval_res['pa-mpjpe'])
            uposed_mesh_error.append(eval_res['ume'])
            posed_mesh_error.append(eval_res['pme'])

            if self.global_step % 200 == 0:
                print(f'step:{self.global_step} \t MPJPE:{np.mean(mpjpe)*1000} \t PAMPJPE:{np.mean(pampjpe)*1000}')
        
        # save results
        mpjpe = np.stack(mpjpe)
        pampjpe = np.stack(pampjpe)
        uposed_mesh_error = np.stack(uposed_mesh_error)
        posed_mesh_error = np.stack(posed_mesh_error)
        np.save(osp.join(self.exppath, 'mpjpe'), mpjpe)
        np.save(osp.join(self.exppath, 'pampjpe'), pampjpe)
        np.save(osp.join(self.exppath, 'ume'), uposed_mesh_error)
        np.save(osp.join(self.exppath, 'pme'), posed_mesh_error)
        print("== Final Results ==")
        print('MPJPE:', np.mean(mpjpe)*1000)
        print('PAMPJPE:', np.mean(pampjpe)*1000)
        print('Mesh Error:', uposed_mesh_error.mean(), posed_mesh_error.mean())
        with open(osp.join(self.exppath, 'performance.txt'), 'w') as f:
            _res = f'MPJPE:{mpjpe.mean()*1000}, PAMPJPE:{pampjpe.mean()*1000}, ume:{uposed_mesh_error.mean()}, pme:{posed_mesh_error.mean()}'
            f.write(_res)

    def meta_adaptation(self, unlabeled_batch, labeled_batch):
        learner = self.model.clone()
        lowerlevel_loss, lowerlevel_lossdict = self.lowerlevel_adaptation(learner, unlabeled_batch, labeled_batch)
        learner.adapt(lowerlevel_loss)
        upperlevel_loss, upperlevel_lossdict = self.upperlevel_adaptation(learner, unlabeled_batch, labeled_batch)
        self.write_summaries(lowerlevel_lossdict, self.global_step)
        self.write_summaries(upperlevel_lossdict, self.global_step)
        return upperlevel_loss

    def lowerlevel_adaptation(self, learner, unlabeled_batch, labeled_batch):
        losses_dict = {}
        if self.options.dataset_name == '3dpw':
            uimage, us2d = unlabeled_batch['image'], unlabeled_batch['smpl_j2ds']
        elif self.options.dataset_name == 'mpi-inf-3dhp':
            uimage, us2d = unlabeled_batch['image'], unlabeled_batch['keypoint']
        # record history
        self.history_info[self.global_step] = {'image': uimage.clone().detach().cpu(), 's2d': us2d.clone().detach().cpu()}
        
        # forward
        pred_rotmat, pred_betas, pred_cam = learner(uimage)
        smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True, pose2rot=False)
        pred_s3d = smpl_out['s3d']
        pred_vts = smpl_out['vts']

        # calculate 2d kp loss
        s2dloss = cal_s2d_loss(pred_s3d, us2d, pred_cam)
        # cal prior loss
        shape_prior_loss = shape_prior(pred_betas)
        pose_prior_losses = pose_prior(pred_rotmat, pred_betas, gmm_prior=True)
        gmm_prior_loss = pose_prior_losses['gmm']
        total_loss = s2dloss * self.options.s2dloss_weight +\
                     gmm_prior_loss * self.options.gmmpriorloss_weight +\
                     shape_prior_loss * self.options.shapeloss_weight
        losses_dict['lowlevel_unlabeled/s2dloss'] = s2dloss
        losses_dict['lowlevel_unlabeled/gmmloss'] = gmm_prior_loss
        losses_dict['lowlevel_unlabeled/shapeloss'] = shape_prior_loss
        losses_dict['lowlevel_unlabeled/unlabeled_loss'] = total_loss

        # update on labeled batch
        labeled_loss, labeled_s3d_loss, labeled_s2d_loss, labeled_loss_pose, labeled_loss_beta = self.adapt_on_labeled_data(learner, labeled_batch)
        total_loss += labeled_loss * self.options.labelloss_weight
        losses_dict['lowlevel_labeled/s3dloss'] = labeled_s3d_loss
        losses_dict['lowlevel_labeled/s2dloss'] = labeled_s2d_loss
        losses_dict['lowlevel_labeled/thetaloss'] = labeled_loss_pose
        losses_dict['lowlevel_labeled/betaloss'] = labeled_loss_beta
        losses_dict['lowlevel_labeled/labeled_loss'] = labeled_loss
        return total_loss, losses_dict

    def upperlevel_adaptation(self, learner, unlabeled_batch, labeled_batch):
        losses_dict = {}
        if self.options.dataset_name == '3dpw':
            uimage, us2d = unlabeled_batch['image'], unlabeled_batch['smpl_j2ds']
        elif self.options.dataset_name == 'mpi-inf-3dhp':
            uimage, us2d = unlabeled_batch['image'], unlabeled_batch['keypoint']
        # get history
        histories = self.get_history()
        history_image, history_us2d = histories['image'], histories['s2d']
        
        # forward
        pred_rotmat, pred_betas, pred_cam = learner(uimage)
        smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam,neutral=True, pose2rot=False)
        pred_s3d = smpl_out['s3d']
        pred_vts = smpl_out['vts']

        # calculate 2d kp loss
        s2dloss = cal_s2d_loss(pred_s3d, us2d, pred_cam)
        # cal prior loss
        shape_prior_loss = shape_prior(pred_betas)
        pose_prior_losses = pose_prior(pred_rotmat, pred_betas, gmm_prior=True)
        gmm_prior_loss = pose_prior_losses['gmm']
        total_loss = s2dloss * self.options.s2dloss_weight +\
                     gmm_prior_loss * self.options.gmmpriorloss_weight +\
                     shape_prior_loss * self.options.shapeloss_weight
        losses_dict['upperlevel_unlabeled/s2dloss'] = s2dloss
        losses_dict['upperlevel_unlabeled/gmmloss'] = gmm_prior_loss
        losses_dict['upperlevel_unlabeled/shapeloss'] = shape_prior_loss
        losses_dict['upperlevel_unlabeled/singleframe_loss'] = total_loss
        
        # mean teacher loss
        ema_rotmat, ema_betas, ema_cam = self.ema_model(uimage)
        consistent_losses = cal_consistent_constrain(pred_rotmat, pred_betas, pred_cam, ema_rotmat, ema_betas, ema_cam)
        s3dloss_mt, s2dloss_mt, poseloss_mt, betaloss_mt = consistent_losses['s3dloss'], consistent_losses['s2dloss'], consistent_losses['poseloss'], consistent_losses['betaloss']
        consistent_loss = s3dloss_mt * self.options.consistent_s3d_weight + s2dloss_mt * self.options.consistent_s2d_weight + \
                                poseloss_mt * self.options.consistent_pose_weight + betaloss_mt * self.options.consistent_beta_weight
        total_loss += consistent_loss * self.options.consistentloss_weight
        losses_dict['upperlevel_unlabeled/s3dloss_mt'] = s3dloss_mt
        losses_dict['upperlevel_unlabeled/s2dloss_mt'] = s2dloss_mt
        losses_dict['upperlevel_unlabeled/poseloss_mt'] = poseloss_mt
        losses_dict['upperlevel_unlabeled/betaloss_mt'] = betaloss_mt
        losses_dict['upperlevel_unlabeled/consistentloss'] = consistent_loss

        labeled_loss, labeled_s3d_loss, labeled_s2d_loss, labeled_loss_pose, labeled_loss_beta = self.adapt_on_labeled_data(learner, labeled_batch)
        total_loss += labeled_loss * self.options.labelloss_weight
        losses_dict['upperlevel_labeled/s3dloss'] = labeled_s3d_loss
        losses_dict['upperlevel_labeled/s2dloss'] = labeled_s2d_loss
        losses_dict['upperlevel_labeled/thetaloss'] = labeled_loss_pose
        losses_dict['upperlevel_labeled/betaloss'] = labeled_loss_beta
        losses_dict['upperlevel_labeled/labeled_loss'] = labeled_loss

        if history_image is not None:
            # motion loss
            pred_history_rotmat, pred_history_betas, pred_history_cam = learner(history_image)
            history_smpl_out = decode_smpl_params(pred_history_rotmat, pred_history_betas, pred_history_cam, neutral=True, pose2rot=False)
            pred_history_s3d = history_smpl_out['s3d']
            motion_loss = cal_motion_loss(pred_s3d, pred_cam, pred_history_s3d, pred_history_cam, us2d, history_us2d)
            losses_dict['upperlevel_unlabeled/motionloss'] = motion_loss
            total_loss += motion_loss * self.options.motionloss_weight
        return total_loss, losses_dict

    def adapt_on_labeled_data(self, learner, databatch):
        image = databatch['img'].squeeze(0)
        trg_s3d = databatch['pose_3d'].squeeze(0)
        trg_s2d = databatch['keypoints'].squeeze(0)
        trg_betas = databatch['betas'].squeeze(0)
        trg_pose = databatch['pose'].squeeze(0)
        losses_dict = {}
        pred_rotmat, pred_betas, pred_cam = learner(image)
        smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True, pose2rot=False)
        pred_s3d = smpl_out['s3d']
        pred_vts = smpl_out['vts']

        s2d_loss = cal_s2d_loss(pred_s3d, trg_s2d, pred_cam)
        s3d_loss = cal_s3d_loss(pred_s3d, trg_s3d)
        trg_rotmat = batch_rodrigues(trg_pose.view(-1, 3)).view(-1, 24, 3, 3)
        loss_pose = F.mse_loss(pred_rotmat, trg_rotmat)
        loss_beta = F.mse_loss(pred_betas, trg_betas)
        loss = s3d_loss * 5 + s2d_loss * 5 + loss_pose * 1 + loss_beta * 0.001
        return loss, s3d_loss, s2d_loss, loss_pose, loss_beta
    
    def test(self, databatch, joint_mapper_gt, joint_mapper_h36m):
        if '3dpw' in self.options.dataset_name:
            gt_pose = databatch['pose']
            gt_betas = databatch['betas']
            gender = databatch['gender']            
        
        with torch.no_grad():
            # forward
            self.model.eval()
            images = databatch['image']
            pred_rotmat, pred_betas, pred_cam = self.model(images)
            pred_smpl_out = decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']

            # calculate metrics 
            J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to(self.device)
            # get 14 gt joints
            if 'h36m' in self.options.dataset_name or 'mpi-inf' in self.options.dataset_name:
                gt_keypoints_3d = databatch['oripose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices_female = smpl_female(global_orient=gt_pose[:, :3], body_pose=gt_pose[:, 3:], betas=gt_betas).vertices
                gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
                # get unposed mesh
                t_rotmat = torch.eye(3,3).unsqueeze(0).unsqueeze(0).repeat(pred_rotmat.shape[0], pred_rotmat.shape[1], 1, 1).to(self.device)
                pred_smpl_out = decode_smpl_params(t_rotmat, pred_betas, pred_cam, neutral=True)
                unposed_pred_vts = pred_smpl_out['vts']
                unposed_gt_vertices = smpl_male(global_orient=t_rotmat[:,1:], body_pose=t_rotmat[:,0].unsqueeze(1), betas=gt_betas, pose2rot=False).vertices 
                unposed_gt_vertices_female = smpl_female(global_orient=t_rotmat[:,1:], body_pose=t_rotmat[:,0].unsqueeze(1), betas=gt_betas, pose2rot=False).vertices 
                unposed_gt_vertices[gender==1, :, :] = unposed_gt_vertices_female[gender==1, :, :]

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d_orig = torch.matmul(J_regressor_batch, pred_vts)
            pred_pelvis = pred_keypoints_3d_orig[:, [0], :].clone()
            pred_keypoints_3d_orig = pred_keypoints_3d_orig[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d_orig - pred_pelvis

            cached_results = {'imgname': databatch['imgname'][0],
                                'img_process': databatch['img_process'],
                                #'p_id': p_id.cpu().numpy(),
                                'verts': pred_vts.cpu().numpy(),
                                #'pred_cam_t': pred_cam_t.cpu().numpy(),
                                'pred_cam': pred_cam.cpu().numpy(),
                                #'bbox': batch['bbox'],
                                'rotmat': pred_rotmat.cpu().numpy(),
                                'beta': pred_betas.cpu().numpy(),
                                'gt_keypoints_3d': gt_keypoints_3d.cpu().numpy(),
                                'pred_keypoints_3d_orig': pred_keypoints_3d_orig.cpu().numpy(),
                                'pred_keypoints_3d': pred_keypoints_3d.cpu().numpy()}


            # Render mesh to image
            self.render_output_mesh(cached_results['img_process'], cached_results['verts'], cached_results['pred_cam'], pred_pelvis.detach().cpu().numpy())

            # Save mesh render result
            seq_name = databatch['imgname'][0].split('/')[1]
            image_id = databatch['imgname'][0].split('/')[-1].split('image_')[-1].split('.')[0]
            
            """
            if not os.path.exists(os.path.join(self.exppath, 'result', seq_name)):
                os.makedirs(os.path.join(self.exppath, 'result', seq_name))

            joblib.dump(cached_results, osp.join(self.exppath, 'front', seq_name, f'image_{image_id}.pt'))
            """



            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vts)
            pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
            # 1. MPJPE
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            # 2. PA-MPJPE
            r_error, pck_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(),needpck=True, reduction=None)
            results = {'mpjpe': error, 'pa-mpjpe': r_error, 'pck': pck_error}
            if '3dpw' in self.options.dataset_name:
                # 3. shape evaluation
                unposed_mesh_error = torch.sqrt(((unposed_gt_vertices - unposed_pred_vts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                posed_mesh_error = torch.sqrt(((gt_vertices - pred_vts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                results['ume'] = unposed_mesh_error
                results['pme'] = posed_mesh_error
        return results

    def get_history(self,):
        history_idx = self.global_step - self.options.prev_n
        if history_idx > 0:
            hist_uimage, hist_us2d = self.history_info[history_idx]['image'].to(self.device), \
                                     self.history_info[history_idx]['s2d'].to(self.device)
        else:
            hist_uimage, hist_us2d = None, None
        return {'image': hist_uimage, 's2d': hist_us2d}


    def render_output_mesh(self, img_info, verts, pred_cam, pred_pelvis):
        # Render info
        render_result_path = './render_result_openpose'
        render_color = np.array([135, 108, 250])
        gray_render_color = np.array([220, 220, 220])

        # Data loading
        imgname = img_info['imgname'][0]
        img_id = int(imgname.split('/')[-1].split('.jpg')[0].split('image_')[-1])
        theta = img_info['theta']
        beta = img_info['beta']
        s2d = img_info['s2d']
        s2d_smpl = img_info['s2d_smpl']
        center = img_info['center'][0]
        scale = img_info['scale'][0]
        p_id = img_info['p_id'][0].item()

        seq_name = imgname.split('/')[-2] + '_' + str(p_id)

        pred_verts = verts
        pred_verts_root = pred_verts - pred_pelvis
        pred_cam = pred_cam
        
        # Rendering
        oriimg = cv2.imread(os.path.join(config.PW3D_ROOT, imgname))
        ori_h, ori_w = oriimg.shape[:2]
        bbox = np.array([[center[0].item(), center[1].item(), 200*scale]])
        ori_pred_cams = convert_crop_cam_to_orig_img(pred_cam, bbox, ori_w, ori_h)
        # Front
        renderer = Renderer(resolution=(ori_w, ori_h), orig_img=True, wireframe=False)
        rendered_image = renderer.render(oriimg, pred_verts[0], ori_pred_cams[0], color=render_color/255, mesh_filename='demo.obj')
        gray_rendered_image = renderer.render(oriimg, pred_verts[0], ori_pred_cams[0], color=gray_render_color/255, mesh_filename='demo.obj')
        # Side
        white_background = 255 * np.ones((ori_h,ori_w,3), np.uint8)
        black_background = np.zeros((ori_h,ori_w,3), np.uint8)
        side_rendered_image = renderer.render(white_background, pred_verts[0], ori_pred_cams[0], color=render_color/255, mesh_filename='demo.obj', angle=-90, axis=[0,1,0])
        gray_side_rendered_image = renderer.render(white_background, pred_verts[0], ori_pred_cams[0], color=gray_render_color/255, mesh_filename='demo.obj', angle=-90, axis=[0,1,0])

        # Front view
        front_view_path = os.path.join(render_result_path, 'front', seq_name)
        gray_front_view_path = os.path.join(render_result_path, 'front_gray', seq_name)
        if not os.path.exists(front_view_path):
            os.makedirs(front_view_path)
        if not os.path.exists(gray_front_view_path):
            os.makedirs(gray_front_view_path)
        cv2.imwrite(os.path.join(front_view_path, f'{img_id:05d}.png'), rendered_image)
        cv2.imwrite(os.path.join(gray_front_view_path, f'{img_id:05d}.png'), gray_rendered_image)

        # Side view
        side_view_path = os.path.join(render_result_path, 'side', seq_name)
        gray_side_view_path = os.path.join(render_result_path, 'side_gray', seq_name)
        if not os.path.exists(side_view_path):
            os.makedirs(side_view_path)
        if not os.path.exists(gray_side_view_path):
            os.makedirs(gray_side_view_path)
        cv2.imwrite(os.path.join(side_view_path, f'{img_id:05d}.png'), side_rendered_image)
        cv2.imwrite(os.path.join(gray_side_view_path, f'{img_id:05d}.png'), gray_side_rendered_image)

        # Mesh obj
        obj_path = os.path.join(render_result_path, 'obj', seq_name)
        if not os.path.exists(obj_path):
            os.makedirs(obj_path)
        self.save_obj(pred_verts_root[0], smpl_neutral.faces, osp.join(obj_path, f'{img_id:05d}.obj'))
        

    
    def save_obj(self, v, f=None, file_name=''):
        obj_file = open(file_name, 'w')
        for i in range(len(v)):
            obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
        if f is not None:
            for i in range(len(f)):
                obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
        obj_file.close()


    def save_results(self, vts, cam_trans, images_lr, images_hr, p_id, name, bbox, prefix=None): # images_hr: [1. 448, 448, 3]
        use_entire_img = True
        vts = vts.clone().detach().cpu().numpy()
        cam_trans = cam_trans.clone().detach().cpu().numpy()
        images = images_lr.clone().detach()
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        images = np.transpose(images.cpu().numpy(), (0,2,3,1))
        # hr image
        images_hr = images_hr.clone().detach()
        images_hr = images_hr * torch.tensor([0.229, 0.224, 0.225], device=images_hr.device).reshape(1,3,1,1)
        images_hr = images_hr + torch.tensor([0.485, 0.456, 0.406], device=images_hr.device).reshape(1,3,1,1)
        images_hr = np.transpose(images_hr.cpu().numpy(), (0,2,3,1))
        for i in range(vts.shape[0]):
            bbox = bbox.cpu().numpy()
            seq_name = name[i].split('/')[-2] + "_" + str(int(p_id))
            # Check whether this is the end of global step
            if self.global_step == 0:
                self.prev_global_step = self.global_step
            else:
                if self.prev_global_step != self.global_step:
                    if use_entire_img is True:
                        oriimg = cv2.imread(os.path.join(self.imgdir, name[i]))
                        ori_h, ori_w = oriimg.shape[:2]
                        ori_pred_cams = convert_crop_cam_to_orig_img(cam_trans, bbox, ori_w, ori_h)
                        renderer = Renderer(resolution=(ori_w, ori_h), orig_img=True, wireframe=False)
                        rendered_image = renderer.render(oriimg, vts[i], ori_pred_cams[i], color=np.array([254,189,248])/255, mesh_filename='demo.obj')
                        bbox_for_crop = [max(0, int(bbox[0][1]-(bbox[0][2]*1.3/2))), int(bbox[0][1]+(bbox[0][2]*1.3/2)), max(0, int(bbox[0][0]-(bbox[0][2]*1.3/2))), int(bbox[0][0]+(bbox[0][2]*1.3/2))]
                        rendered_image = rendered_image[bbox_for_crop[0]:bbox_for_crop[1], bbox_for_crop[2]:bbox_for_crop[3], :]

                        if not os.path.exists(os.path.join(self.exppath, 'image', seq_name)):
                            os.makedirs(os.path.join(self.exppath, 'image', seq_name), exist_ok=True)
                            self.seq_count[seq_name] = 0

                        # Front view mesh
                        try:
                            cv2.imwrite(osp.join(self.exppath, 'image', seq_name, f'{prefix}_{self.seq_count[seq_name]+i}.png'), rendered_image)
                        except:
                            import pdb; pdb.set_trace()

                        # Side view mesh
                        white_background = 255 * np.ones((ori_h,ori_w,3), np.uint8)
                        side_view_rendered_image = renderer.render(white_background, vts[i], ori_pred_cams[i], color=np.array([254,189,248])/255, mesh_filename='demo.obj', angle=-90, axis=[0,1,0])
                        side_view_rendered_image = side_view_rendered_image[bbox_for_crop[0]:bbox_for_crop[1], bbox_for_crop[2]:bbox_for_crop[3], :]
                        cv2.imwrite(osp.join(self.exppath, 'image', seq_name, f'{prefix}_{self.seq_count[seq_name]+i}_side.png'), side_view_rendered_image)

                        # Save obj file
                        save_obj(vts[i]*1000, smpl_neutral.face, osp.join(self.exppath, 'image', seq_name, f'{prefix}_{self.seq_count[seq_name]+i}.obj'))

                    self.seq_count[seq_name] += 1





if __name__ == '__main__':
    options = parser.parse_args()
    exppath = osp.join(options.expdir, options.name)
    os.makedirs(exppath)
    argsDict = options.__dict__
    with open(f'{exppath}/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    adaptor = Adaptator(options)
    adaptor.inference()
