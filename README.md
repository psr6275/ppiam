# Privacy-preserving Inference Resistant to Model Extraction Attacks
This repository is the official implementation of "Privacy-preserving Inference Resistant to Model Extraction Attacks"
## Requirements
For training, attack and evaluation (python emulation for HE and MPC),
```setup
pip install -r requirements.txt
```
For Crypten,
```setup2
pip install -r crypten_requirements.txt
```
For HEAAN, our evaluation code is provided in "HENN/", where the run code is included in "HENN/run/". We use the public library in https://github.com/snucrypto/HEAAN.git. Thus, our codes for HE evaluation are written in C++.

## Pre-trained Models
We provide the pre-trained target models in "results/" directory. 
- target network (MPC-MNIST): "mnist_orig_oe.pth"
- target network (HE-MNIST): "mnist_HE_orig_oe.pth"
- target network (MPC-CIFAR10): "cifar_orig_oe.pth"
- target network (HE-CIFAR10): "cifar_HE_orig_oe.pth"
Similarly, we provide 8 fake networks for AM and IAM (which includes swd in the model name). 

## Evaluation
To evaluate the models against model extraction attack,
```eval
cd codes
python main.py --help
python main.py --dataset "cifar10" --exp 1
```
To evaluate the model performance using Crypten,
```eval2
cd codes
python crypten_main.py --help
python crypten_main.py --dataset "mnist" --exp 2 --tau 0.9
```

## Others
To check the adjustable setting, 
```help
cd codes
python train.py --help
python attack.py --help
python evaluate.py --help
python crypten_evaluate.py --help
```
