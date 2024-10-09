# Flower-Image-Classifier
In this project, an image classifier is built to recognize different species of flowers.
The project is broken down into three main steps
#1 Load and preprocess the image dataset
#2 Train the image classifier on the dataset
#3 Use the trained classifier to predict image content

##Important modules for this project include
# Imports here
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#import torchvision
#import torch.nn.functional as F
#from torch.autograd import Variable
#import numpy as np
#import matplotlib.pyplot as plt
#import time
#import torch
#from collections import OrderedDict
#from torch import nn
#from torch import optim
#import torch.nn.functional as F
#from torchvision import datasets, transforms, models
#from PIL import Image
#import matplotlib.pyplot as plt
#import numpy as np
#from torch import nn
#import torch.utils.data 
#import pandas as pd
#import helper
import seaborn as sns

#Data Description
The dataset is split into three parts, training, validation, and testing. For the training, transformations such as random scaling, cropping, and flipping were applies which helped the network generalize leading to better performance. Data input was resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets were used to measure the model's performance on data it hasn't seen yet. For these there was no scaling or rotation transformations, but resizing and cropping of the images to the appropriate size.

The pre-trained networks used were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets normalization of the means and standard deviations of the images to were performed. For the means, it was [0.485, 0.456, 0.406] and for the standard deviations it was [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.


