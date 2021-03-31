# BOA - Bilevel Online Adaptation
Code repository for the paper:  
**Bilevel Online Adaptation for Out-of-Domain Human Mesh Reconstruction**  
Shanyan Guan\*, [Jingwei Xu\*](https://scholar.google.com/citations?user=7oepLUoAAAAJ&hl=en), [Yunbo Wang<sup>#</sup>](http://people.csail.mit.edu/yunbo/), Bingbing Ni, Xiaokang Yang  
CVPR 2021  
[[Paper](https://arxiv.org/pdf/2103.16449.pdf)] [[project page](https://sites.google.com/view/humanmeshboa)] [[Supp](https://jbox.sjtu.edu.cn/l/qFD0qN)]

![demo](demo.gif)

## Citation
If you find this code useful for your research or the use data generated by our method, please consider citing this paper:
```
@inproceedings{syguan2021boa,
  Title          = {Bilevel Online Adaptation for Out-of-Domain Human Mesh Reconstruction},
  Author         = {Shanyan, Guan and Jingwei, Xu and Yunbo, Wang and Bingbing, Ni and Xiaokang, Yang},
  Booktitle      = {CVPR},
  Year           = {2021}
}
```

## Requirements
1. Please run `pip install requirements.txt` to install all dependencies.
2. Downloading data related to SMPL: 
* Download the SMPL model and then remove the dependency on Chumpy follwing this [instruction](https://github.com/vchoutas/smplx/tree/master/tools). Then, put the processed models to `data/smpl/`. 
* Download [3rd party files](http://visiondata.cis.upenn.edu/spin/data.tar.gz) which is provided by [SPIN](https://github.com/nkolot/SPIN). Then extact the file and put them to `data/spin_data`.

## Get Started
Download [the base model](https://jbox.sjtu.edu.cn/l/qFDRC4) pre-trained on Human 3.6M. Run the following commond to excute Bilevel online optimization.
```
python boa.py --name boa-1
```

## Preparing Dataset
Before running the BOA, we should process the datasets first. 

- **3DPW**  
Note that this is the guideline to get data according to the *#PS* protocol (i.e. processing 3DPW following SPIN). To obtain data according to the *#PH* protocol, please run the scripts in [HMMR](https://github.com/akanazawa/human_dynamics), and save the results.
  1. Download the [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) dataset. Then edit `PW3D_ROOT` in the `config.py`.
  2. Run the processing script:
     ```
     python process_data.py --dataset 3dpw
     ```

- **MPI-INF-3DHP**  
  1. Download the [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) dataset. Then edit `MPI_INF_3DHP_ROOT` in the `config.py`
  2. Extracting frames by running 
     ```
     cd utils/data_preprocess
     python extract_3dhp_frames.py
     ```
  3. Run the processing script: 
     ```
     python process_data.py --dataset 3dhp
     ```
     Note that installing spacepy is required. Please refer to this [website](https://spacepy.github.io/install_linux.html) to install it.

- **Human 3.6M**  
**Mesh annotation is a necessary file**. However I cannot provide it in public. If you get access to it, following the next intruction to process Human 3.6M.
  1. Download Human 3.6M. [Downloader](https://github.com/kotaro-inoue/human3.6m_downloader) is suggested.
  2. Unpack files: 
     ```
     
     python unpack_h36m.py
     ```
     And then edit `H36M_ROOT` in the `config.py`.
  3. Check if the mesh annotations need to be rectified:
     ```
     cd utils/data_preprocess
     python check_mosh.py
     ```
     If the joints is not aligned to the image, please rectify them by
     ```
     python rectify_pose.py
     ```
  3. Processing Human 3.6M.
     `python process_data.py --dataset h36m`

## To be fixed
When I try the code on the 3080 GPU with Pytorch 1.8 and CUDA 11.1, the adaptation speed is slower more than 10 times. I will fix this issue ASAP. 

## References
Here are some great resources we benefit:
- [SPIN](https://github.com/nkolot/SPIN)
- [learn2learn](https://github.com/learnables/learn2learn)
- [VIBE](https://github.com/mkocabas/VIBE)
- [SMPL-X](https://github.com/vchoutas/smplx/tree/master/tools)
