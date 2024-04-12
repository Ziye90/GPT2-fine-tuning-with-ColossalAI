#     Resnet-with-ColossalAI

Assignment 6: Reproduce one Colossal-AI example

Objective: The goal of this assignment is to gain hands-on experience with Colossal-AI, a powerful tool for distributed training of large-scale AI models.

#     Model
Resnet18

#   The dataset
CIFAR10 consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.
# Instructions on how to run your code


### Set up in colab 
* Mount to google drive
```
from google.colab import drive
drive.mount('/content/drive')
```

* Install requirements
```
pip install -r requirements.txt
```
* train with torch DDP with fp32
```
!colossalai run --nproc_per_node 1 train.py -c ./ckpt-fp32 
```
 The makeenv.ipynb file shows the details of how to set up the enviroment, train, evaluate and run the codeon google Colab

# Experiment results
| Model     | Booster DDP with FP32 | Booster DDP with FP16 | 
| --------- |-----------------------|-----------------------|
| ResNet-18 | 41.00 %               | 45.53 %               | 
