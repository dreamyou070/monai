import os
import torch.nn.functional as F
import torch
from PIL import Image
import torch
import numpy as np

gt_path = '003.png'
gt_img = np.array(Image.open(gt_path).convert('L').resize((64,64)))
gt_torch = torch.tensor(gt_img)/255
print(gt_torch.shape)