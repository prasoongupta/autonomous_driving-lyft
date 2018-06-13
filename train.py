# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:45:45 2018

@author: prasoon
"""
import os
import tensorflow as tf
import numpy as np
from helper import  get_datagens
from easydict import EasyDict
import yaml
from model import Segmentation_Model

config = EasyDict(yaml.load(open('myconfig.yml','r')))
slim = tf.contrib.slim

seed = 0

batch_size = config.train.batch_size
val_batch_size = config.train.val_batch_size

learning_rate = config.train.learning_rate
num_epochs = config.train.num_epochs

restore_model = config.train.restore_model
pre_train = config.train.pre_train

model_name = config.train.model
model_checkpoint_dir = config.train.checkpoint_dir


tf.reset_default_graph()

config_proto = tf.ConfigProto(device_count={'GPU':0})
#sess = tf.Session(config=config_proto)
sess = tf.Session()
tf.set_random_seed(seed)


    
seg_model = Segmentation_Model(config)

tf_image_pl, tf_labels_pl, tf_logits, tf_loss, tf_preds = seg_model.get_model()


global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(tf_loss, global_step=global_step)    

graph = tf.get_default_graph()

base_init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess.run(base_init_op)
[op.name for op in graph.get_operations() if 'resnet' not in op.name and 'gradient' not in op.name
 and 'save' not in op.name ]
if pre_train:
    restore_vars = [v for v in slim.get_model_variables() if 'logit' not in v.name ]
    print('restoring from imagenet wts...')
    pretrain_init_fn = slim.assign_from_checkpoint_fn('./resnet/resnet_v1_101.ckpt', var_list=restore_vars)
    pretrain_init_fn(sess)
            
if restore_model:
    print('restoring model...')
    saver = tf.train.Saver(tf.trainable_variables()) 
    checkpoint_file = tf.train.latest_checkpoint(model_checkpoint_dir)
    saver.restore(sess,checkpoint_file)
    print('restored -',checkpoint_file)
    '''
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in tf.global_variables()])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
    '''
    
saver = tf.train.Saver()

train_datagen_fn, val_datagen_fn = get_datagens()

def test_eval(datagen):
    total_loss = 0
    samples_ctr = 0
    for (batch_images, batch_labels) in datagen:        
        feed_dict = {tf_image_pl:batch_images, tf_labels_pl:batch_labels}
        val_loss, preds = sess.run([tf_loss, tf_preds], feed_dict=feed_dict)
        total_loss += val_loss
        samples_ctr += 1
    return total_loss / float(samples_ctr)
    

for epoch in range(num_epochs):
    steps = 0
    eloss = 0
    train_datagen = train_datagen_fn(batch_size=batch_size)
    for (batch_images, batch_labels) in train_datagen:
        feed_dict = {tf_image_pl:batch_images, tf_labels_pl:batch_labels}
        _, loss, = sess.run([train_op, tf_loss], feed_dict=feed_dict)
        steps += 1
        eloss += loss
        if steps%100==0:
                print('epoch: {} step {} loss: {:.3f}'.format(epoch,steps, eloss/100))
                eloss = 0
    print('epoch: {} loss: {:.2f}'.format(epoch,loss))
    val_datagen = val_datagen_fn(batch_size=val_batch_size)
    val_loss = test_eval(val_datagen)
    print('epoch: {} Validation loss: {:.3f}'.format(epoch,val_loss))
    
saver.save(sess,model_checkpoint_dir+'/model.ckpt',global_step=10000)

