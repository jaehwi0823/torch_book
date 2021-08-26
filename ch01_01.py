import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
%matplotlib inline
import torch
import torchvision
from torchvision import models, transforms

# print(torch.__version__)
# print(torchvision.__version__)

# pre-trained model
net = models.vgg16(pretrained=True)
net.eval()

# pre-processing
class BaseTransform():
    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    def __call__(self, img):
        return self.base_transform(img)

img_path = './data/goldenretriever-3724972_640.jpg'
img = Image.open(img_path)
plt.imshow(img)
img.show()

# transform test
tran = BaseTransform(resize=224,
                     mean = (0.485, 0.456, 0.406),
                     std = (0.229, 0.224, 0.225))
img_tran = tran(img)
img_tran = torch.permute(img_tran,(1,2,0))
img_tran = torch.clip(img_tran, 0, 1)
plt.imshow(img_tran)
plt.show()

# post-precessing
