import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

random.seed(21)
np.random.seed(21)

data_path = "D:\\brats2024\\data\\data_processed"
train_and_test_ids = os.listdir(data_path)

train_ids, val_test_ids = train_test_split(train_and_test_ids, test_size=0.1, random_state=21) # seed = 21
#val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5,random_state=21)  # seed = 21
print("Using {} images for training, {} images for validation.".format(len(train_ids),len(val_test_ids)))

with open('D:\\brats2024\\unet\\train_split.txt','w') as f:
    f.write('\n'.join(train_ids))

with open('D:\\brats2024\\unet\\valid_split.txt','w') as f:
    f.write('\n'.join(val_test_ids))

#with open('/scratch/yc6785/csdn_brats24/test_split.txt','w') as f:
    #f.write('\n'.join(test_ids))
