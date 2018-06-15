# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:10:22 2018

@author: prasoon
"""
import cv2
import tensorflow as tf
#from helper import data_generator, get_image_paths, get_datagens
#from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
#import time
import sys, json, base64 #, skvideo.io, 
import numpy as np
#from PIL import Image
#from io import BytesIO, StringIO
from easydict import EasyDict
import yaml

model = 'resnet'

config = EasyDict(yaml.load(open('myconfig.yml', 'r')))


frozen_graph = os.path.join(config.main.base_dir, config.main.frozen_graph)


print('Frozen graph file', frozen_graph)

tf.reset_default_graph()
restored_graph_def = tf.GraphDef()
restored_graph_def.ParseFromString(tf.gfile.GFile(frozen_graph,'rb').read())

tf.import_graph_def(restored_graph_def,name='')
graph = tf.get_default_graph()
sess = tf.Session(graph=graph)

graph = tf.get_default_graph()

tf_input_image = graph.get_tensor_by_name('placeholders/inputs:0')
tf_preds = graph.get_tensor_by_name('predictions/preds:0')


image_shape = (384,384)

mean=[0.485, 0.456, 0.406] 
std=[0.229, 0.224, 0.225]


def get_predictions(image):
    feed_dict = {tf_input_image:image}
    return sess.run(tf_preds, feed_dict=feed_dict)

file = sys.argv[-1]
file = 'predictions/test_video.mp4'

# Define encoder function
def encode(array):
    reval, buffer = cv2.imencode('.png',array)
    return base64.b64encode(buffer).decode("utf-8")

def get_overlay(image, label):
    mask = np.zeros_like(image)
    road = np.where(label==1)
    car = np.where(label==2)
    mask[road] = (0,255,0)
    mask[car] = (255,0,0)
    result = cv2.addWeighted(image,1,mask,0.3,0)
    return result

#video = skvideo.io.vread(file)
video = cv2.VideoCapture(file)
answer_key = {}

# Frame numbering starts at 1
frame = 1
flag, rgb_frame = video.read()

frame_height = int(video.get(4))
frame_width = int(video.get(3))
out_name = 'predictions/output_video.mp4'
out_codec = cv2.VideoWriter_fourcc(*'MP4V')
fps = 500
out_video = cv2.VideoWriter(out_name, out_codec, fps, (frame_width, frame_height),True)
while(flag):
#for flag,rgb_frame in video.read():
    #rgb_frame = video[0]
    final_img = rgb_frame
    orig_shape = final_img.shape    
    #final_img = cv2.resize(rgb_frame, (image_shape[1],image_shape[0]), interpolation=cv2.INTER_AREA)
    #norm_image = cv2.normalize(final_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #norm_image = norm_image - mean
    #norm_image = norm_image/ std
    
    preds = get_predictions(np.expand_dims(final_img, axis=0))    
    
    #preds = np.reshape(preds,(image_shape[0],image_shape[1],3))
    preds = np.squeeze(np.argmax(preds,axis=-1))

    preds = cv2.resize(preds, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

    binary_car_result = np.where(preds==2,1,0).astype('uint8')    
    # Look for road :)
    binary_road_result = np.where(preds==1,1,0).astype('uint8')
    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    #color_frame = cv2.cvtColor(preds, cv2.COLOR_GRAY2RGB)
    result = get_overlay(rgb_frame, preds)
    out_video.write(result)
    # Increment frame
    frame+=1
    flag, rgb_frame = video.read()

video.release()
out_video.release()
# Print output in proper json format
with open('predictions/inference.json','w') as outfile:
    outfile.write(json.dumps(answer_key))
#print(json.dumps(answer_key))