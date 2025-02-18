import os.path as osp


SMPL_MODEL_DIR = 'data/smpl'
SMPL_MEAN_PARAMS = 'data/spin_data/smpl_mean_params.npz'

PW3D_ROOT = '/mnt/disk3/danieljung0121/3DPW_Original'
MPI_INF_3DHP_ROOT = '/mnt/disk3/danieljung0121/MPI_INF_3DHP'
H36M_ROOT = '/mnt/disk1/danieljung0121/Human36M'

# folder to save processed files
DATASET_NPZ_PATH = 'data/dataset_extras'
PW3D_ANNOT_DIR = osp.join(DATASET_NPZ_PATH, '3dpw_vid')

JOINT_REGRESSOR_TRAIN_EXTRA = 'data/spin_data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/spin_data/J_regressor_h36m.npy'

# Path to test/train npz files
DATASET_FILES = [ {'h36m': osp.join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'mpi-inf-3dhp': osp.join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz'),
                   '3dpw': osp.join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                  },

                  {'h36m': osp.join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   '3dpw': osp.join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                   'mpi-inf-3dhp': osp.join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_test.npz')
                  }
                ]
