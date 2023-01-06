# LeNet-5

**Pytorch** implementation of **LeNet-5**，my first step to implement artificial neural networks using pytorch.



# Results

The trained LeNet-5 model reach a **98.69%** accuracy on MNIST test dataset on my computer. 

When I used the trained model to recognize my own handwritten numbers, the accuracy could not reach 98%. I'm trying to figure out the underlying mechanism of this. If your have any idea, welcome to discuss with me.



# Requirements

```
torch==1.13.1
torchvision==0.14.1
opencv==4.6.0
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

train the model on MNIST-train-dataset

```
python main.py --mode train
```

test the model MNIST-test-dataset

```
python main.py --mode test
```



# Test your own handwriting numbers

Put your own handwriting number pictures in the folder **./data/test_pic/hand_write/**, and now you can test your own handwriting numbers using the model trained on MNIST-train-dataset

```
python main.py --mode test --test_mode custom
```

