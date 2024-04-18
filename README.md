# Deep Learning for Low-light Vision





## 📢  Foreword

This repository is a collection of methods for low-light vision tasks, encompassing techniques for vision quality-driven low-light image enhancement and advanced low-light visual algorithms driven by quality-based recognition. It comprises a curated list of papers, code, and datasets relevant to low-light vision.




## 📜 Table of Contents




- [Deep Learning for Low-light Vision](#deep-learning-for-low-light-vision)
  - [📢  Foreword](#--foreword)
  - [📜 Table of Contents](#-table-of-contents)
  - [📬 Datasets](#-datasets)
    - [Paired Dataset](#paired-dataset)
    - [Unpaired Dataset](#unpaired-dataset)
  - [⏳ Review and Benchmark](#-review-and-benchmark)
  - [🪐 Low-light Image Enhancement Methods](#-low-light-image-enhancement-methods)
    - [Deep Learning-based Methods](#deep-learning-based-methods)
  - [Non-deep Learning-based Methods](#non-deep-learning-based-methods)
    - [HE-based Methods](#he-based-methods)
    - [Statistical Model-based Methods](#statistical-model-based-methods)
    - [Traditional Retinex-based Methods](#traditional-retinex-based-methods)
  - [🚂 Dark Object Detection](#-dark-object-detection)
  - [🪂 Other High-level Low-light Vision Tasks](#-other-high-level-low-light-vision-tasks)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Object Tracking](#object-tracking)
    - [Human Pose Estimation](#human-pose-estimation)
  - [✍ Metrics](#-metrics)
  - [📡 Acknowledgement](#-acknowledgement)


<span id="datasets"></span>
## 📬 Datasets
### Paired Dataset

|Year|Datasets|Numbers|Format|Synthetic/Real|Annotations|Usage|Link|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2011|<div style="width: 85pt"> MIT-Adobe FiveK|5000|Raw Image|Real|-|Train/Test|[Paper](http://people.csail.mit.edu/vladb/photoadjust/db_imageadjust.pdf) [Dataset](https://data.csail.mit.edu/graphics/fivek/)|
|2016|SID|5094|Raw Image|Real|-|Train/Test|[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Learning_to_See_CVPR_2018_paper.html) [Dataset](https://github.com/cchen156/Learning-to-See-in-the-Dark)|
|2018|LOL|500|RGB Image|Synthetic+Real|-|Train/Test|[Paper](https://arxiv.org/abs/1808.04560) [Dataset](https://daooshee.github.io/BMVC2018website/)|
|2018|SCIE|4413|RGB Image|Real|-|Train/Test|[Paper](https://ieeexplore.ieee.org/abstract/document/8259342/) [Dataset](https://github.com/csjcai/SICE)|
|2019|SMOID|179|Raw Video|Real|-|Train/Test|[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Jiang_Learning_to_See_Moving_Objects_in_the_Dark_ICCV_2019_paper.html) [Dataset](https://github.com/MichaelHYJiang) |
|2019|DRV|202|Raw videos|Real|-|Train/Test|[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Chen_Seeing_Motion_in_the_Dark_ICCV_2019_paper.html)[Dataset](https://github.com/cchen156/Seeing-Motion-in-the-Dark) |
|2021|VE-LOL-L|2500|RGB Image|Synthetic+Real|-|Train/Test|[Paper](https://link.springer.com/article/10.1007/s11263-020-01418-8) [Dataset](https://flyywh.github.io/IJCV2021LowLight_VELOL/)|
|2021|SDSD|150|RGB Video|Real|-|Train/Test|[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Seeing_Dynamic_Scene_in_the_Dark_A_High-Quality_Video_Dataset_ICCV_2021_paper.html) [Dataset](https://github.com/dvlab-research/SDSD) |
|2021|ACDC|4006|RGB Image|Real|19-class sematic labels|Train/Val|[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Seeing_Dynamic_Scene_in_the_Dark_A_High-Quality_Video_Dataset_ICCV_2021_paper.html) [Dataset](https://acdc.vision.ee.ethz.ch/)|
|2024|Dark Vision-L|13455|RGB Image|Real|15-class bounding boxes|Train/Test|[Paper](https://arxiv.org/abs/2301.06269) [Dataset](https://arxiv.org/abs/2301.06269)|
|2024|Dark Vision-H|32|RGB Video|Real|4-class bounding boxes|Train/Test|[Paper](https://arxiv.org/abs/2301.06269) [Dataset](https://arxiv.org/abs/2301.06269)|


### Unpaired Dataset

|Year|Datasets|Numbers|Format|Synthetic/Real|Annotations|Usage|Link|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|2012|DICM|64|RGB Image|Real|-|Test|[Paper](https://ieeexplore.ieee.org/abstract/document/6467022) [Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|2013|NPE|84|RGB Image|Real|-|Test|[Paper](https://ieeexplore.ieee.org/abstract/document/6512558) [Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|2015|MEF|17|RGB Image|Real|-|Test|[Paper](https://ieeexplore.ieee.org/abstract/document/7120119) [Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|2017|LIME|10|RGB Image|Real|-|Test|[Paper](https://ieeexplore.ieee.org/abstract/document/7782813) [Dataset](https://drive.google.com/file/d/0BwVzAzXoqrSXb3prWUV1YzBjZzg/view)|
|2017|VV|24|RGB Image|Real|-|Test|[Paper](https://link.springer.com/article/10.1007/s11042-017-4783-x) [Dataset](https://drive.google.com/drive/folders/1lp6m5JE3kf3M66Dicbx5wSnvhxt90V4T)|
|2019|Dark Zurich|151|RGB video|Real|19-class sematic labels|Train/Val|[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.html) [Dataset](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)|
|2019|ExDARK|7363|RGB Image|Real|<div style="width: 117pt"> 12-class bounding boxes|Train/Test|[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314218304296) [Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)|
|2020|DARK FACE|10000|RGB Image|Real|Face bounding box|Train/Val|[Paper](https://ieeexplore.ieee.org/abstract/document/9049390) [Dataset](https://flyywh.github.io/CVPRW2019LowLight/)|
|2021|LLVIP|33672|RGB+Infrared Image|Real|1-class bounding box|Train/Test|[Paper](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/html/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.html) [Dataset](https://github.com/bupt-ai-cz/LLVIP)|
|2021|VE-LOL-H|10940|RGB Image|Real|Face bounding box|Train|[Paper](https://link.springer.com/article/10.1007/s11263-020-01418-8) [Dataset](https://flyywh.github.io/IJCV2021LowLight_VELOL/)|
|2021|LLIV-Phone|45148|RGB Image|Real|-|Test|[Paper](https://ieeexplore.ieee.org/abstract/document/9609683) [Dataset](https://drive.google.com/file/d/1QS4FgT5aTQNYy-eHZ_A89rLoZgx_iysR/view)|


<span id="review-and-benchmark"></span>
## ⏳ Review and Benchmark

| Year | Publication      | Title | Link                                                         | 
| :--: |:---------:| :---------------------------:| :------------------------------------------------------------:| 
| 2021 | IJCV      | Benchmarking Low-Light Image Enhancement and Beyond          | [Paper](https://link.springer.com/article/10.1007%2Fs11263-020-01418-8) |            |
| 2021 | IEEE PAMI | Low-Light Image and Video Enhancement Using Deep Learning: A Survey | [Paper](https://doi.org/10.1109/TPAMI.2021.3126387) [Web]((https://drive.google.com/file/d/1QS4FgT5aTQNYy-eHZ_A89rLoZgx_iysR/view))           |            |
| 2022 | ArXiv     | Low-Light Image and Video Enhancement: A Comprehensive Survey and Beyond | [Paper](http://arxiv.org/abs/2212.10772) [Web](https://github.com/shenzheng2000/llie_survey) |            |
| 2023 | ArXiv     | DarkVision: A Benchmark for Low-Light Image/Video Perception | [Paper](https://arxiv.org/abs/2301.06269)                      | DarkVision |
| 2023 | Signal Process. | A comprehensive experiment-based review of low-light image enhancement methods and benchmarking low-light image quality assessment  | [Paper](https://linkinghub.elsevier.com/retrieve/pii/S0165168422003607) |            |


<span id="low-light-image-enhancement-methods"></span>
## 🪐 Low-light Image Enhancement Methods

### Deep Learning-based Methods

| Year | Publication | Title|  Abbreviation |Learning|Link |
|:---:|:---:|---|:---:|:---:|:---:|
| 2017 | PR      | LLNet: A deep autoencoder approach to natural low-light image enhancement  | LLNet                |Supervised learning | [Paper](https://doi.org/10.1016/j.patcog.2016.06.008)|        
| 2017 | TOG     | Deep bilateral learning for real-time image enhancement      | HDRNet               |Supervised learning |[Paper](https://arxiv.org/abs/1707.02880) [Code](https://github.com/google/hdrnet) | 
| 2018 | BMVC                    | Deep Retinex Decomposition for Low-Light Enhancement         |  Retinex-Net          |Supervised learning|[Paper](https://arxiv.org/abs/1808.04560) [Code](https://github.com/weichen582/RetinexNet) |
| 2018 | BMVC                    | MBLLEN: Low-light Image/Video Enhancement Using CNNs         | MBLLEN               |Supervised learning |[Paper](http://bmvc2018.org/contents/papers/0700.pdf)  [Code](https://github.com/Lvfeifan/MBLLEN) |
| 2018 | PRL | LightenNet: A Convolutional Neural Network for weakly illuminated image enhancement          | LightenNet           |Supervised learning| [Paper](https://doi.org/10.1016/j.patrec.2018.01.010) |
| 2018 | CVPR                    | Learning to See in the Dark                                  | -                   |Supervised learning | [Paper](https://cchen156.github.io/paper/18CVPR_SID.pdf)  [code](https://github.com/cchen156/Learning-to-See-in-the-Dark.git) |
| 2018 | TIP                | Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images | SICE                 |Supervised learning| [Paper](https://doi.org/10.1109/TIP.2018.2794218) [Code](https://github.com/csjcai/SICE) |
| 2018 | NeurIPS                 | DeepExposure: Learning to expose photos with asynchronously reinforced adversarial learning       | DeepExposure| Reinforcement learning              | [Paper](https://doi.org/10.1145/3181974) [Code](https://github.com/yuanming-hu/exposure) |
| 2019 | TIP                | DeepISP: Towards Learning an End-to-End Image Processing Pipeline    | DeepISP              |Supervised learning| [Paper](https://ieeexplore.ieee.org/abstract/document/8478390)                   |
| 2019 |  TIP                | Low-Light Image Enhancement via a Deep Hybrid Network                 | -                   |Supervised learning | [Paper](https://ieeexplore.ieee.org/document/8692732)|
| 2019 | ACM MM                  | Kindling the Darkness: A Practical Low-light Image Enhancer  | KinD                 |Supervised learning| [Paper](https://dl.acm.org/doi/abs/10.1145/3343031.3350926) [Code](https://github.com/zhangyhuaee/KinD) |
| 2021 | IJCV                 | Beyond brightening low-light images  | KinD++                 |Supervised learning| [Paper](https://link.springer.com/article/10.1007/s11263-020-01407-x) [Code](https://github.com/zhangyhuaee/KinD_plus) |
| 2019 | CVPR                    | Underexposed Photo Enhancement Using Deep Illumination Estimation | DeepUPE              |Supervised learning| [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953588) [Code](https://github.com/wangruixing/DeepUPE) |
| 2019 | ACM MM                    | Zero-shot restoration of back-lit images using deep internal learning  | ExCNet              |Zero-shot learning| [Paper](https://dl.acm.org/doi/abs/10.1145/3343031.3351069) [Code](https://cslinzhang.github.io/ExCNet/)|
| 2020 | CVPR                    | Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement  | Zero-DCE             |Zero-shot learning| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)  [Code](https://github.com/Li-Chongyi/Zero-DCE)|
| 2021 | TPAMI                   | Learning to enhance low-light image via zero-reference deep curve estimation | Zero-DCE++  |Zero-shot learning| [Paper](https://ieeexplore.ieee.org/abstract/document/9369102) [Code](https://github.com/Li-Chongyi/Zero-DCE_extension/) |
| 2020 | AAAI                    | EEMEFN: Low-light image enhancement via edge-enhanced multi-exposure fusion network  | EEMEFN  |Supervised learning| [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/7013) |
| 2020 | TIP                    | Lightening network for low-light image enhancement  | DLN  |Supervised learning| [Paper](https://ieeexplore.ieee.org/abstract/document/9141197) |
| 2020 | TMM                    | Luminance-aware pyramid network for low-light image enhancement  | LPNet  |Supervised learning| [Paper](https://ieeexplore.ieee.org/abstract/document/9186194) |
| 2020 | ECCV                   | Low light video enhancement using synthetic data produced with an intermediate domain mapping  | SIDGAN  |Supervised learning| [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58601-0_7) |
| 2020 | ICME                   | Zero-shot restoration of underexposed images via robust retinex decomposition | RRDNet  |Zero-shot learning| [Paper](https://ieeexplore.ieee.org/abstract/document/9102962) [Code](https://aaaaangel.github.io/RRDNet-Homepage/) |
| 2020 | CVPR                   | Learning to Restore Low-Light Images via Decomposition-and-Enhancement  | FIDE  |Supervised learning| [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Xu_Learning_to_Restore_Low-Light_Images_via_Decomposition-and-Enhancement_CVPR_2020_paper.html) |
| 2020 | TIP                   | Towards Unsupervised Deep Image Enhancement With Generative Adversarial Network  | UEGAN  |Unsupervised learning| [Paper](https://ieeexplore.ieee.org/abstract/document/9204448) [Code](https://github.com/eezkni/UEGAN) |
| 2020 | CVPR                    | From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement  | DRBN                 |Semi-supervised learning| [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.pdf)  [Code](https://github.com/flyywh/CVPR-2020-Semi-Low-Light/blob/master)|
| 2021 | TIP                | EnlightenGAN: Deep Light Enhancement without Paired Supervision  |Unsupervised learning| EnlightenGAN    | [Paper](https://ieeexplore.ieee.org/abstract/document/9334429) [Code](https://github.com/TAMU-VITA/EnlightenGAN) |    
| 2021 | TIP                   | Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement  | Sparse  |Supervised learning| [Paper](https://ieeexplore.ieee.org/abstract/document/9328179) |
| 2021 | TCSVT                   | RetinexDIP: A unified deep framework for low-light image enhancement| RetinexDIP  |Zero-shot learning| [Paper](https://ieeexplore.ieee.org/document/9405649) [Code](https://github.com/zhaozunjin/RetinexDIP) |
| 2021 | CVPR                    | Retinex-Inspired Unrolling with Cooperative Prior Architecture Search for Low-Light Image Enhancement | RUAS                 |Zero-shot learning| [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf) [Code](https://github.com/dut-media-lab/RUAS) |
| 2023 | TPAMI | Learning With Nested Scene Modeling and Cooperative Architecture Search for Low-Light Vision | RUAS-plus   |Zero-shot learning| [Paper](https://ieeexplore.ieee.org/document/9914672/) [Code](https://github.com/vis-opt-group/ruas) |
| 2021 | CVPR                    | Nighttime Visibility Enhancement by Increasing the Dynamic Range and Suppression of Light Effects | -               |Semi-supervised learning| [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sharma_Nighttime_Visibility_Enhancement_by_Increasing_the_Dynamic_Range_and_Suppression_CVPR_2021_paper.html)|
| 2022 | PAMI               | Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time  | 3DLUT |Semi-supervised learning| [Paper](https://ieeexplore.ieee.org/document/9206076) [Code](https://github.com/HuiZeng/Image-Adaptive-3DLUT)|
| 2022 | TNNLS               | Learning Deep Context-Sensitive Decomposition for Low-Light Image Enhancement | CSDNet |Semi-supervised learning| [Paper](https://ieeexplore.ieee.org/document/9420270) [code](https://github.com/KarelZhang/CSDNet-CSDGAN) |
| 2022 | JVCIR                   | R2RNet: Low-Light Image Enhancement via Real-Low to Real-Normal Network | R2RNet               |Supervised learning | [Paper](https://www.sciencedirect.com/science/article/pii/S1047320322002322) [Code](https://github.com/abcdef2000/R2RNet)|
| 2022 | CVPR                    | Toward Fast, Flexible, and Robust Low-Light Image Enhancement  | SCI          |Zero-shot learning| [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Toward_Fast_Flexible_and_Robust_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [Code](https://github.com/vis-opt-group/SCI)|
| 2022 | AAAI                    | Low-Light Image Enhancement with Normalizing Flow             |LLFlow      |Supervised learning | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20162) [Code](https://github.com/wyf0912/LLFlow)|
| 2022 | BMCV                    | You only need 90K parameters to adapt light: a light weight transformer for image enhancement and exposure correction | IAT  |Supervised learning| [Paper](https://bmvc2022.mpi-inf.mpg.de/0238.pdf) [Code](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) |
| 2022 |  TNNLS  | DRLIE: Flexible Low-Light Image Enhancement via Disentangled Representations| DRLIE                     |Unsupervised learning | [Paper](https://ieeexplore.ieee.org/document/9833451/) |
| 2022 | CVPR                    | URetinex-Net: Retinex-Based Deep Unfolding Network for Low-Light Image Enhancement  | URetinex-Net         |Supervised learning| [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_URetinex-Net_Retinex-Based_Deep_Unfolding_Network_for_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [Code](https://github.com/AndersonYong/URetinex-Net)|
| 2022 | CVPR                    | SNR-Aware Low-Light Image Enhancement                         |     SNR          |Supervised learning| [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html) [Code](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance)|
| 2022 | ECCV                    | Unsupervised Night Image Enhancement: When Layer Decomposition Meets Light-Effects Suppression                        |     -        |Unsupervised learning | [Paper](https://link.springer.com/chapter/10.1007/978-3-031-19836-6_23) [Code](https://github.com/jinyeying/night-enhancement)|
| 2022 | ACM MM                    | Cycle-Interactive Generative Adversarial Network for Robust Unsupervised Low-Light Enhancement                        |     CIGAN          |Unsupervised learning | [Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548006)|
| 2022 | WACV                    | Semantic-Guided Zero-Shot Learning for Low-Light Image/Video Enhancement  | SGZ          |Zero-shot learning| [Paper](https://openaccess.thecvf.com/content/WACV2022W/RWS/html/Zheng_Semantic-Guided_Zero-Shot_Learning_for_Low-Light_ImageVideo_Enhancement_WACVW_2022_paper.html) [Code](https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement)|
| 2023 | ICCV  | Retinexformer: One-stage retinex-based transformer for low-light image enhancement|   Retinexformer    |Supervised learning | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Retinexformer_One-stage_Retinex-based_Transformer_for_Low-light_Image_Enhancement_ICCV_2023_paper.html) [Code](https://github.com/caiyuanhao1998/Retinexformer) |
| 2023 | CVPR                    | Learning a simple low-light image enhancer from paired low-light instances  | PairLIE |Unsupervised learning| [Paper](http://openaccess.thecvf.com/content/CVPR2023/html/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.html) [Code](https://github.com/zhenqifu/pairlie)|
| 2023 |  TNNLS | Retinex Image Enhancement Based on Sequential Decomposition With a Plug-and-Play Framework   | -  |Zero-shot learning| [Paper](https://ieeexplore.ieee.org/document/10144685)|
| 2023 |  TIP | Unsupervised Low-Light Video Enhancement with Spatial-Temporal Co-attention Transformer  | LightenFormer  |Unsupervised learning| [Paper](https://ieeexplore.ieee.org/document/10210621/) |
| 2023 | ACM MM | CLE Diffusion: Controllable Light Enhancement Diffusion Model  | CLE Diffusion  |Unsupervised learning| [Paper](https://arxiv.org/abs/2308.06725) [Code](https://github.com/YuyangYin/CLEDiffusion)|
| 2023 |  TNNLS | Low-Light Image Enhancement by Retinex-Based Algorithm Unrolling and Adjustment | DecNet  |Supervised learning| [Paper](https://ieeexplore.ieee.org/document/10174279) [Code](https://github.com/Xinyil256/RAUNA2023) |
| 2023 | ICCV  | ExposureDiffusion: Learning to expose for low-light image enhancement  |     ExposureDiffusion     |Supervised learning| [Paper](https://arxiv.org/pdf/2307.07710.pdf) [Code](https://github.com/wyf0912/ExposureDiffusion)|
| 2023 | TPAMI  | ExposureDiffusion: Learning to expose for low-light image enhancement |     MIRNet-v2     |Supervised learning| [Paper](https://arxiv.org/pdf/2307.07710.pdf) [Code](https://github.com/swz30/MIRNetv2) |
| 2023 |  TMM  | Cycle-Retinex: Unpaired Low-Light Image Enhancement via Retinex-Inline CycleGAN|      Cycle-Retinex     |Unsupervised learning| [Paper](https://ieeexplore.ieee.org/document/10130403) [Code](https://github.com/mummmml/Cycle-Retinex) |
| 2024 | CVPR  | Towards Robust Event-guided Low-Light Image Enhancement: A Large-Scale Real-World Event-Image Dataset and Novel Approach |     EvLight     |Supervised learning| [Paper](https://arxiv.org/pdf/2404.00834.pdf) [Code](https://github.com/EthanLiang99/EvLight) |
| 2024 | CVPR  | Zero-Reference Low-Light Enhancement via Physical Quadruple Priors |     -     |Zero-shot learning| [Paper](https://arxiv.org/pdf/2403.12933.pdf) [Code](https://github.com/daooshee/QuadPrior/)|
| 2024 |  PR | A non-regularization self-supervised Retinex approach to low-light image enhancement with parameterized illumination estimation    |- |Zero-shot  learning| [Paper](https://www.sciencedirect.com/science/article/pii/S0031320323007227) [Code](https://github.com/zhaozunjin/NeurBR) |
| 2024 |  TCSVT  | Low-Light Image Enhancement With Multi-Scale Attention and Frequency-Domain Optimization  |     -   |Supervised learning| [Paper](https://ieeexplore.ieee.org/document/10244055)|
| 2024 |  T-ITS  | Double Domain Guided Real-Time Low-Light Image Enhancement for Ultra-High-Definition Transportation  Surveillance  |     DDNet   |Supervised learning| [Paper](https://ieeexplore.ieee.org/abstract/document/10423894) [Code](https://github.com/QuJX/DDNet)|
| 2024 | CVPR | Binarized Low-light Raw Video Enhancement  |     -   |Supervised learning| [Paper](https://arxiv.org/abs/2403.19944) [Code](https://github.com/ying-fu/BRVE)|


## Non-deep Learning-based Methods

### HE-based Methods


| Year | Publication       | Title                                                     |       Abbreviation                                                   |  Link |
| :--: | :--------:| ----- | :--------: | :-----: |
| 1990 | TCE | Contrast limited adaptive histogram equalization: speed and effectiveness  | CLAHE | [Paper](https://ieeexplore.ieee.org/document/109340?arnumber=109340)|
| 2007 | TCE | Brightness Preserving Dynamic Histogram Equalization for Image Contrast Enhancement | BPDHE | [Paper](https://ieeexplore.ieee.org/document/4429280) [Code](codes/bpdhe.m) |
| 2007 | TCE | A Dynamic Histogram Equalization for Image Contrast Enhancement | DHE   |[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4266947) | 
| 2007 | TCE | Fast Image/Video Contrast Enhancement Based on Weighted Thresholded Histogram Equalization |  WTHE  |[Paper](https://ieeexplore.ieee.org/document/4266969?arnumber=4266969) |
| 2011 | TIP | Contextual and Variational Contrast Enhancement              |  CVC   |[Paper](http://ieeexplore.ieee.org/abstract/document/5773086/) |
| 2013 | TIP | Contrast enhancement based on layered difference representation of 2D histograms | LDR   |[Paper](http://mcl.korea.ac.kr/projects/LDR/2013_tip_cwlee_final_hq.pdf)  [Web](http://mcl.korea.ac.kr/cwlee_tip2013/) | 
| 2013 | ICASSP   | High efficient contrast enhancement using parametric approximation | POHE  |[Paper](http://150.162.46.34:8080/icassp2013/pdfs/0002444.pdf) | 



### Statistical Model-based Methods
| Year | Publication       | Title                                                     |       Abbreviation                                                   |  Link |
| :--: | :--------:| ----- | :--------: | :-----: |
| 2016 | ICIP | Hue-preserving perceptual contrast enhancement  | - | [Paper](https://ieeexplore.ieee.org/abstract/document/7533124)|
| 2016 | TVCG | Underexposed Video Enhancement via Perception-Driven Progressive Fusion  | - | [Paper](https://ieeexplore.ieee.org/abstract/document/7167723)|
| 2016 | TIP | Contrast Enhancement by Nonlinear Diffusion Filtering  | - | [Paper](https://ieeexplore.ieee.org/abstract/document/7352346)|
| 2018 | ICIP | Restoration of Unevenly Illuminated Images  | - | [Paper](https://ieeexplore.ieee.org/abstract/document/8451278)|
| 2017 | ICCV | A New Low-Light Image Enhancement Algorithm Using Camera Response Model  | - | [Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w43/html/Ying_A_New_Low-Light_ICCV_2017_paper.html)|


### Traditional Retinex-based Methods

| Year |  Publication                     | Title                                                        |  Abbreviation                                                        | Link  |
| ---- | :----------------------:| ------------------------------------------------------------ | :------------------------------------------------------------: | :-----: |
|2003 | IJCV               | A Variational Framework for Retinex|-  | [Paper](https://link.springer.com/article/10.1023/A:1022314423998)           | 
| 2013 |TIP               | Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images |NPE   | [Paper](https://ieeexplore.ieee.org/document/6512558) [Web](https://shuhangwang.wordpress.com/2015/12/14/naturalness-preserved-enhancement-algorithm-for-non-uniform-illumination-images/) [Code](https://www.dropbox.com/s/096l3uy9vowgs4r/Code.rar) | 
| 2015 |TIP               | A Probabilistic Method for Image Enhancement With Simultaneous Illumination and Reflectance Estimation |SRIE  | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7229296) [Code](codes/PM_SIRE.zip) | 
| 2016 | CVPR                   | A Weighted Variational Model for Simultaneous Reflectance and Illumination Estimation| SRIE  | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Fu_A_Weighted_Variational_CVPR_2016_paper.pdf)  [Code](codes/WV_SIRE.zip)  |
| 2016 | SP      | A fusion-based enhancing method for weakly illuminated images | MF    |[Paper](https://doi.org/10.1016/j.sigpro.2016.05.031) [Code](codes/MF.rar) | 
| 2016 | ACM MM                 | LIME: A Method for Low-light Image Enhancement               | LIME  | [Paper](https://arxiv.org/pdf/1605.05034.pdf ) [Web](https://sites.google.com/view/xjguo/lime) [Code](https://drive.google.com/file/d/0BwVzAzXoqrSXb3prWUV1YzBjZzg/view) |
| 2017 | TIP               | LIME: Low-Light Image Enhancement via Illumination Map Estimation |LIME  | [Paper](http://ieeexplore.ieee.org/document/7782813/) [Code](https://github.com/estija/LIME) | 
|


<span id="dark-object-detection"></span>
## 🚂 Dark Object Detection

| Year | Publication | Title|  Abbreviation |Type|Link |
|:---:|:---:|---|:---:|:---:|:---:|
| 2019 |  ICCVW  | FuseMODNet: Real-Time Camera and LiDAR Based Moving Object Detection for Robust Low-Light Autonomous Driving |     -   |Multimodal fusion| [Paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/ADW/Rashed_FuseMODNet_Real-Time_Camera_and_LiDAR_Based_Moving_Object_Detection_for_ICCVW_2019_paper.html)|
| 2020 |  ECCV  | YOLO in the Dark - Domain Adaptation Method for Merging Multiple Models |     Dark -YOLO   |Domain adaptation | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58589-1_21)|
| 2020 |  IEEE Access  | Making of Night Vision: Object Detection Under Low-Illumination |    NVD  |Feature fusion  | [Paper](https://ieeexplore.ieee.org/abstract/document/9134757)|
| 2021 |  ICCVW | Adapting Deep Neural Networks for Pedestrian-Detection to Low-Light Conditions Without Re-Training | -  |Preprocessing (non-end-to-end)  | [Paper](https://openaccess.thecvf.com/content/ICCV2021W/TradiCV/html/Shah_Adapting_Deep_Neural_Networks_for_Pedestrian-Detection_to_Low-Light_Conditions_Without_ICCVW_2021_paper.html)|
| 2021 |  TII |Low-Illumination Image Enhancement for Night-Time UAV Pedestrian Detection | -  |Preprocessing (non-end-to-end)  | [Paper](https://ieeexplore.ieee.org/abstract/document/9204832)|
| 2021 |  ICCVW |Single-Stage Face Detection Under Extremely Low-Light Conditions | MSRCR   |Preprocessing (end-to-end)  | [Paper](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/html/Yu_Single-Stage_Face_Detection_Under_Extremely_Low-Light_Conditions_ICCVW_2021_paper.html)|
| 2021 | ICCV  |Multitask AET With Orthogonal Tangent Regularity for Dark Object Detection| MAET   |Image darkening | [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.html) [Code](https://github.com/cuiziteng/ICCV_MAET) |
| 2021 |  BMVC  |Crafting Object Detection in Very Low Light| -   |Image darkening | [Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0085.pdf) [Code](https://github.com/ying-fu/LODDataset)|
| 2021 |  ACML |Domain Adaptive YOLO for One-Stage Cross-Domain Detection| DA-YOLO  |Domain adaptation| [Paper](https://proceedings.mlr.press/v157/zhang21c.html)|
| 2021 |  ICIP |Multiscale Domain Adaptive Yolo For Cross-Domain Object Detection| MS-DAYOLO  |Domain adaptation| [Paper](https://ieeexplore.ieee.org/abstract/document/9506039)|
| 2021 |CVPR |HLA-Face: Joint High-Low Adaptation for Low Light Face Detection|HLA |Domain adaptation| [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_HLA-Face_Joint_High-Low_Adaptation_for_Low_Light_Face_Detection_CVPR_2021_paper.html) |
| 2022 |  AAAI |Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions| IA-YOLO  |Preprocessing (end-to-end)| [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20072) [Code](https://github.com/wenyyu/Image-Adaptive-YOLO)|
| 2022 | ACCV |DENet: Detection-driven Enhancement Network for Object Detection under Adverse Weather Conditions| DE-YOLO  |Preprocessing (end-to-end)| [Paper](https://openaccess.thecvf.com/content/ACCV2022/html/Qin_DENet_Detection-driven_Enhancement_Network_for_Object_Detection_under_Adverse_Weather_ACCV_2022_paper.html)|
| 2022 | ACM MM |PIA: Parallel Architecture with Illumination Allocator for Joint Enhancement and Detection in Low-Light|PIA  |Feature fusion| [Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548041)|
| 2022 |TMM |Recurrent Exposure Generation for Low-Light Face Detection|REGDet  |Feature fusion| [Paper](https://ieeexplore.ieee.org/abstract/document/9387154)|
| 2022 |CVPR |Target-Aware Dual Adversarial Learning and a Multi-Scenario Multi-Modality Benchmark To Fuse Infrared and Visible for Object Detection|TarDAL |Multimodal fusion| [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.html) [Code](https://github.com/dlut-dimt/TarDAL )|
| 2023 |FI |A lightweight object detection network in low-light conditions based on depthwise separable pyramid network and attention mechanism on embedded platforms|DS-PyLENet |Preprocessing (end-to-end)| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0016003223001047) |
| 2023 |Information Fusion |A complementary dual-backbone transformer extracting and fusing weak cues for object detection in extremely dark videos|DVD-TR |Feature fusion | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253523001380) |
| 2023 |ICCV |FeatEnHancer: Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision|FeatEnHancer |Feature fusion | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Hashmi_FeatEnHancer_Enhancing_Hierarchical_Features_for_Object_Detection_and_Beyond_Under_ICCV_2023_paper.html) |
| 2023 |TNNLS  |Image Enhancement Guided Object Detection in Visually Degraded Scenes|- |Feature fusion | [Paper](https://ieeexplore.ieee.org/abstract/document/10130799) |
| 2023 |TPAMI  |Unsupervised Face Detection in the Dark|HLAv2 |Domain adaptation| [Paper](https://ieeexplore.ieee.org/abstract/document/9716838) |
| 2023 |TNNLS  |LRAF-Net: Long-Range Attention Fusion Network for Visible–Infrared Object Detection|LRAF-Net |Multimodal fusion| [Paper](https://ieeexplore.ieee.org/abstract/document/10144688) |
| 2024 |CVPR |Boosting Object Detection with Zero-Shot Day-Night Domain Adaptation|DAI-Net |Domain adaptation | [Paper](https://arxiv.org/abs/2312.01220) |


<span id="other-high-level-low-light-vision-tasks"></span>
## 🪂 Other High-level Low-light Vision Tasks
### Semantic Segmentation
| Year | Publication | Title|  Abbreviation |Link |
|:---:|:---:|---|:---:|:---:|
| 2019 |ICCV  |Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation|GCMA| [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Sakaridis_Guided_Curriculum_Model_Adaptation_and_Uncertainty-Aware_Evaluation_for_Semantic_Nighttime_ICCV_2019_paper.html)     [Code](https://github.com/sakaridis/MGCDA) |
| 2020|ACM MM  |Integrating Semantic Segmentation and Retinex Model for Low-Light Image Enhancement|ISSR| [Paper](https://dl.acm.org/doi/abs/10.1145/3394171.3413757)     [Code](https://github.com/XFW-go/ISSR) |
| 2020|TPAMI  |Map-Guided Curriculum Domain Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation|GCMA+| [Paper](https://trace.ethz.ch/publications/2019/GCMA_UIoU/MGCDA_UIoU-Sakaridis+Dai+Van_Gool-IEEE_TPAMI_20.pdf)     [Code](https://github.com/sakaridis/MGCDA) |
| 2021 |ICCV  |ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding|ACDC| [Paper](Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html) [Code](https://acdc.vision.ee.ethz.ch )|
| 2021 |ACM MM |Best of Both Worlds: See and Understand Clearly in the Dark|-| [Paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548259) [Code](https://github.com/k914/contrastive-alternative-learning)|

### Object Tracking
| Year | Publication | Title|  Abbreviation |Link |
|:---:|:---:|---|:---:|:---:|
| 2020 |PRL |Modality-correlation-aware sparse representation for RGB-infrared object tracking|-| [Paper](https://www.sciencedirect.com/science/article/pii/S0167865518307633) |
| 2021 |ICCV  |Object Tracking by Jointly Exploiting Frame and Event Domain|-| [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Object_Tracking_by_Jointly_Exploiting_Frame_and_Event_Domain_ICCV_2021_paper.html)     |
| 2021 |IROS |DarkLighter: Light Up the Darkness for UA V Tracking|DarkLighter| [Paper](https://ieeexplore.ieee.org/abstract/document/9636680)   [Code](https://github.com/vision4robotics/DarkLighter)  |
| 2022 |RAL |Tracker Meets Night: A Transformer Enhancer for UAV Tracking|SCT| [Paper](https://ieeexplore.ieee.org/abstract/document/9696362) [Code](https://github.com/vision4robotics/SCT)    |
| 2022 |IROS |HighlightNet: Highlighting Low-Light Potential Features for Real-Time UAV Tracking|HightlightNet| [Paper](https://ieeexplore.ieee.org/abstract/document/9981070) [Code](https://github.com/vision4robotics/HighlightNet)    |

### Human Pose Estimation
| Year | Publication | Title|  Abbreviation |Link |
|:---:|:---:|---|:---:|:---:|
| 2020 |TIM |An RGB/Infra-Red camera fusion approach for Multi-Person Pose Estimation in low light environments|POISON| [Paper](https://ieeexplore.ieee.org/abstract/document/9290033/) |
| 2022 |ECCV|Image Illumination Enhancement for Construction Worker Pose Estimation in Low-light Conditions|UIRE-Net| [Paper](https://link.springer.com/chapter/10.1007/978-3-031-25082-8_10)     |
| 2023 |CVPR  |Human Pose Estimation in Extremely Low-Light Conditions|ExLPose| [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Human_Pose_Estimation_in_Extremely_Low-Light_Conditions_CVPR_2023_paper.html)  [Web](https://cg.postech.ac.kr/research/ExLPose/)   |


<span id="metrics"></span>
## ✍ Metrics

| Metric               |   Abbreviation   | Full-/Non-Reference        | Link             | 
| :------------------: | :------: | :------------------------:   | :----------------: |
| Peak Signal to Noise Ratio | PSNR | Full-Reference |- |
| Structural Similarity Index Measure | SSIM| Full-Reference | - |
| Mean Absolute Error | MAE | Full-Reference | - |
| Learned Perceptual Image Patch Similarity | LPIPS | Full-Reference | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html)  [Code](https://github.com/richzhang/PerceptualSimilarity) |
| Lightness Order Error | LOE |  Non-Reference | [Paper](https://ieeexplore.ieee.org/document/6512558) [Code](http://blog.sina.com.cn/u/2694868761) |
| Natural Image Quality Evaluator | NIQE  | Non-Reference | [Paper](https://ieeexplore.ieee.org/document/6353522) [Code](http://live.ece.utexas.edu/research/quality/niqe_release.zip)|
| Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision | Q-Bench  | Non-Reference (LLM-driven) | [Paper](https://arxiv.org/abs/2309.14181) [Code](https://github.com/Q-Future/Q-Bench)|
| A Benchmark for Multi-modal Foundation Models on Low-level Vision: from Single Images to Pairs | Q-Bench+ | Non-Reference (LLM-driven) | [Paper](https://arxiv.org/abs/2402.07116) [Code](https://github.com/Q-Future/Q-Bench)|
| Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models |Q-Instruct | Non-Reference (LLM-driven)| [Paper](https://arxiv.org/abs/2311.06783) [Code](https://github.com/Q-Future/Q-Instruct)|





<span id="acknowledgement"></span>
## 📡 Acknowledgement

- [OpenXE](https://github.com/baidut/OpenCE)
- [Awesome-Low-Light-Image-Enhancement](https://github.com/dawnlh/awesome-low-light-image-enhancement)
- [Lighting-the-Darkness-in-the-Deep-Learning-Era-Open](https://github.com/Li-Chongyi/Lighting-the-Darkness-in-the-Deep-Learning-Era-Open)

