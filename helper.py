# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:41:12 2018

@author: prasoon
"""

import os
import shutil
import tensorflow as tf
import requests
import zipfile
from glob import glob
import scipy.misc   
import cv2 
import numpy as np
import random
import matplotlib.pyplot as plt

mean=[0.485, 0.456, 0.406] 
std=[0.229, 0.224, 0.225]
image_shape=(384,384)

def get_image_paths(data_dir='./Train'):

    images_dir = os.path.join(data_dir,'CameraRGB')
    labels_dir = os.path.join(data_dir,'CameraSeg')
    
    image_paths = glob(os.path.join(images_dir,'*.png'))
    label_files = glob(os.path.join(labels_dir,'*.png'))
    
    label_paths= {os.path.join(images_dir,os.path.basename(path)):path for path in label_files}
    return image_paths, label_paths

    
def preprocess_labels(lab):
    lab_new = lab.copy()
    lane_pixels = (lab == 6).nonzero()
    lab_new[lane_pixels] = 7
    vehicle_pixels = (lab == 10).nonzero()
    hood_indices= (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    lab_new[hood_pixels] = 0    
    #lab_new = np.where(np.isin(lab_new,[7,10]),lab_new,0)  
    lab_new = np.where(np.logical_or(lab_new==7,lab_new==10),lab_new,0)   
    lab_new = np.where(lab_new==7,1,lab_new)   
    lab_new = np.where(lab_new==10,2,lab_new)
    
    return lab_new

def data_generator(image_paths, label_paths, augment_flag=False):
    def data_gen_fn(batch_size=8):
        random.shuffle(image_paths)
        for start_index in range(0, len(image_paths), batch_size):
            end_index = start_index + batch_size
            batch_images, batch_labels = [], []
            for image_path in image_paths[start_index:end_index]:
                img = cv2.imread(image_path)
                lab = cv2.imread(label_paths[image_path])[:,:,2]
                lab = preprocess_labels(lab)                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if augment_flag:
                    rgb = augment_image(rgb)
                #final_img = cv2.resize(rgb, (image_shape[1],image_shape[0]), interpolation=cv2.INTER_AREA)
                #norm_image = cv2.normalize(final_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                #norm_image = norm_image - mean
                #norm_image = norm_image/ std
                lab = cv2.resize(lab, (image_shape[1],image_shape[0]), interpolation=cv2.INTER_AREA)            
                batch_images.append(rgb)
                batch_labels.append(lab)
            yield np.array(batch_images), np.array(batch_labels)
    return data_gen_fn

def get_datagens(data_dir='./Train', train_test_split = 0.9):
    image_paths, label_paths = get_image_paths(data_dir)
    random.shuffle(image_paths)
    train_image_paths = image_paths[:int(train_test_split*len(image_paths))]
    val_image_paths = image_paths[int(train_test_split*len(image_paths)):]
    train_gen_fn = data_generator(train_image_paths, label_paths, augment_flag=True)
    val_gen_fn = data_generator(val_image_paths, label_paths)
    return train_gen_fn, val_gen_fn

def augment_image(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    v_change = np.random.randint(-100,100)
    hsv[:,:,2] = hsv[:,:,2] + v_change
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)    
    return img
    

            
"""
image_paths = get_image_paths()
image = image_paths[15]
img = cv2.imread(image)
norm_image = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
rgb = rgb - mean
rgb = rgb / std

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
plt.hist(hsv[:,:,2].reshape(-1))
hsv[:,:,2] = hsv[:,:,2] + 255
hsv = np.clip(hsv,0,255)
img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

contrast = 0
brightness = -0
img = img*(contrast/127 + 1) - contrast + brightness
img = np.clip(img,0,255)
img = np.uint8(img)
plt.imshow(img)

"""

"""
lab = cv2.imread(label_paths[image_path])[:,:,2]
lab = preprocess_labels(lab)

rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
final_img = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_AREA)
lab = cv2.resize(lab, (224,224), interpolation=cv2.INTER_AREA)               
zero_img = np.zeros_like(lab)
road = np.where(lab==1)
car = np.where(lab==2)
mask = np.dstack((zero_img, zero_img, zero_img ))
mask[road] = (0,255,0)
mask[car] = (255,0,0)
result = cv2.addWeighted(final_img,1,mask,0.3,0)

plt.imshow(result)
"""
     
        