# When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation
This is the official implementation of the SIGIR 2023 paper "When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation" based on PyTorch.

[ACM Digital Library](https://doi.org/10.1145/3539618.3591786) This link will be available after the ACM SIGIR 2023.


<!-- Please cite our paper if you use this repository.

```
@inproceedings{si2023sesrec,
  title={When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation},
  author={Zihua Si, Zhongxiang Sun, Xiao Zhang, Jun Xu, Xiaoxue Zang, Yang Song, Kun Gai, Ji-Rong Wen},
  booktitle={Proceedings of the ACM SIGIR Conference 2023},
  pages={2256--2267},
  year={2023}
}
``` -->


## Overview

The main implementation of SESRec can be found in the file `models/SESRec.py`. 
The architecture of SESRec is shown in the following figure:

<img src="./assest/model.png" width="800" height="400">

## Research Questions

We have concluded some frequently asked questions in the file `FAQ.md`.

## Reproduction
Check the following instructions for reproducing experiments.
### Experimental Setting
All the hyper-parameter settings of SESRec on both datasets can be found in files `config/SESRec_commercial.yaml` and `config/SESRec_amazon.yaml`.
The settings of two datasets can be found in file `config/const.py`.



### Dataset
Since the Kuaishou dataset is a proprietary industrial dataset, here we release the ready-to-use data of the Amazon (Kindle Store) dataset. The ready-to-use data can be downloaded from [link](https://drive.google.com/file/d/1HvdhqzKIRbzjMOlXp9j4Hh5KGvX9oTBw/view?usp=sharing).

### Quick Start
#### 1. Download data
Download and unzip data from this [link](https://drive.google.com/file/d/1HvdhqzKIRbzjMOlXp9j4Hh5KGvX9oTBw/view?usp=sharing). Place data files in the folder `data`.

#### 2. Satisfy the requirements
Our experiments were done with the following python packages:
```
python==3.8.13
torch==1.9.0
numpy==1.23.2
pandas==1.4.4
scikit-learn==1.1.2
tqdm==4.64.0
PyYAML==6.0
```

#### 3. Train and evaluate our model:
Run codes in command line:
```bash
python3 main.py --name SESRec --workspace ./workspace/SESRec --gpu_id 0  --epochs 30 --model SESRec  --batch_size 256 --dataset_name amazon
```

#### 4. Check training and evaluation process:
After training, check log files, for example, `workspace/SESRec/log/default.log`.

### Environments

We conducted the experiments based on the following environments:
* CUDA Version: 11.1
* OS: CentOS Linux release 7.4.1708 (Core)
* GPU: The NVIDIA® T4 GPU
* CPU: Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz

### Contact
If you have any questions, feel free to contact us through email zihua_si@ruc.edu.cn or GitHub issues. Thanks!