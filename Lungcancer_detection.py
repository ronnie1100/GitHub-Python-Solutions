#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import WeightedRandomSampler
import pydicom
from pynetdicom import AE, evt
from pynetdicom.sop_class import CTImageStorage, PatientRootQueryRetrieveInformationModelGet


# In[2]:


data = pd.read_csv("metadata.csv")
data


# In[3]:


file_path = "1-002.dcm"
data = pydicom.dcmread(file_path)
print(data)


# In[4]:


file_path = "1-002.dcm"  # Replace 'username' and 'your_dcm_file.dcm' with your actual username and dcm file name
ds = pydicom.dcmread(file_path)

# Rescale the image data
rescale_intercept = ds.RescaleIntercept
rescale_slope = ds.RescaleSlope
pixel_array = ds.pixel_array * rescale_slope + rescale_intercept

# Convert the image data to a NumPy array
image = np.array(pixel_array, dtype=float)

print(image)
plt.imshow(image)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




