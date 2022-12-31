# LeNet-5

**Pytorch** implementation of **LeNet-5**，my first step to implement artificial neural networks using pytorch.



# Requirements

```
torch==1.13.1
torchvision==0.14.1
```



# How to run

clone this repository

```
git clone git@github.com:STU2018/LeNet-5.git
```

 create conda environment

```
conda create -n LeNet_5 python==3.8
conda activate LeNet_5
```

`cd` to the root directory of the repository，and install requirements

```
cd LeNet-5
pip install -r requirement.txt
```

train the model

```
python main.py --mode train
```

test the model

```
python main.py --mode test
```



# Results

The trained LeNet-5 model reach a **98.46%** accuracy on test dataset on my computer. The results may vary slightly on different machines.

