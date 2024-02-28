# <center> Advancements in Dewarping 3D Severely Wrinkled Surfaces to 2D Planes: A Novel Approach Employing the Virtual Flatbed Compressing Method with Coulomb-Hooke-Particle System

by Zhenguo nie, Handing Xu, Yunzhi Chen, Yaguan Li, Yanjie Xu, Meifeng Deng, Levent Burak Kara, Xin-Jun Liu.

<center>**The Virtual Flatbed Compressing Method (VFCM) flattens severely wrinkled surfaces based on the physical simulation
of the flattening process under the pressure of two parallel flatbeds.**

[Project Page]() [arXiv]() [BibTeX]()

![结果图](https://www.z4a.net/images/2023/09/26/result_3.png)</center>

# Environment setup

Clone the repo: `https://github.com/zhenguonie/VFCM-CHPS`

Create python virtualenv through surface.yaml
```
conda env create -f surface.yaml
conda activate surface
pip install plyfile==0.9
pip install sklearn==0.0.post5
```

# Interface

1. Perpare

    Download the pretrained model of LAMA through the web mentioned in `./bin/lama/big-lama/models/pretrained_model_download.txt`, and direct place the downloaded *.ckpt* file in this folder. 

    Place the wrinkled surface in `./ply` folder.

    Place the original flatten surface image in `./img` folder for further evaluation. 

2. Flatten

    Modify the `kwargs['input_pcd']` in `./bin/flatten.py` to your own surface point cloud name in `./ply` and run

    ```
    python ./bin/flatten.py
    ```

    The flattened point cloud and projected image of the surface are saved in `./result` folder, and some temp file will be saved in `./temp` folder. 
    
    If original flatten image is given, the evaluation result can be see in the output panel. 

## Dataset

All the data (severely wrinkled surface point clouds) can be download at [dataset](https://drive.google.com/file/d/1AqtJqT-Xi822LVCtDSVMs5phHN3XH-S6/view?usp=drive_link). 

<!-- # Acknowledgments

LAMA code and models is from [Roman Suvorov](https://github.com/advimman/lama#environment-setup)

SSIM code is from []() -->

# Citation

If you found this code helpful, please consider citing:

```

```
