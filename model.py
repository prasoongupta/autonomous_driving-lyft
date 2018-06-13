# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:52:19 2018

@author: prasoon
"""

import tensorflow as tf
import numpy as np
from network import resnet_v1_101_fcn

class Segmentation_Model():
    def __init__(self, config):
        self._config = config
        self._num_classes = config.model.num_classes
        self._mean = config.model.mean_normalize
        self._image_height = config.model.image_height
        self._image_width = config.model.image_width
        self._class_weights = config.model.class_weights

    def get_model(self):
        
        tf_image_pl, tf_labels_pl = self.get_inputs_pl()
        tf_batch_images = self.tf_preprocess(tf_image_pl)
        tf_img_shape = tf.shape(tf_image_pl)
                
        tf_logits = self.get_logits(tf_batch_images)
        
        tf_preds = self.get_preds(tf_logits,tf_img_shape)
        
        tf_loss = self.get_loss(tf_logits, tf_labels_pl)
        
        return tf_image_pl, tf_labels_pl, tf_logits, tf_loss, tf_preds

    
    def get_inputs_pl(self):
        with tf.name_scope("placeholders"):
            tf_labels = tf.placeholder(tf.int32, shape=(None, None, None), name="labels")
            tf_input_image = tf.placeholder(tf.float32, shape=(None, None, None,3), name="inputs")
            return tf_input_image, tf_labels
    
    def tf_preprocess(self, tf_input_image):
        with tf.name_scope("image_preprocess"):
            tf_resize_imgs = tf.image.resize_images(tf_input_image,(self._image_height,self._image_width))
            tf_batch_images = tf.divide(tf_resize_imgs, 255.0)
            #tf_batch_images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), tf.cast(tf_input_image, tf.float32))
            tf_batch_images  = tf_batch_images - self._mean
            return tf_batch_images
    
    def get_logits(self, tf_batch_images):
        tf_logits_4d = resnet_v1_101_fcn(tf_batch_images, self._num_classes, upsample=8, is_training=True)
        logits = tf.reshape(tf_logits_4d, (-1, self._num_classes), name='logits')
        return logits
    
    def get_loss(self, logits, tf_labels_pl):
        with tf.name_scope("loss"):
            tf_batch_labels = tf.one_hot(tf_labels_pl, self._num_classes)
            tf_batch_labels = tf.reshape(tf_batch_labels, (-1, self._num_classes))    
            
            tf_weights = tf.constant(self._class_weights)
            tf_label_weights = tf.reshape(tf.gather(tf_weights,tf_labels_pl),[-1]) #needs labels before one-hot
            
            tf_wt_loss = tf.losses.softmax_cross_entropy(tf_batch_labels,logits=logits,weights=tf_label_weights)
            
            tf_cross_entropy_loss = tf.reduce_mean(tf_wt_loss)
            tf_regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf_loss = tf_cross_entropy_loss + tf_regularization_loss
            return tf_loss
    
    def get_preds(self, logits, tf_img_shape):
        with tf.name_scope("predictions"):
            tf_preds = tf.nn.softmax(logits)
            tf_preds = tf.reshape(tf_preds,[-1, self._image_height,self._image_width, self._num_classes])
            tf_preds = tf.image.resize_images(tf_preds,(tf_img_shape[1], tf_img_shape[2]))
            tf_preds = tf.identity(tf_preds, name='preds')
            return tf_preds
