# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:33:44 2018

@author: prasoon
"""

import tensorflow as tf
import numpy as np

model = 'resnet'
model_checkpoint_dir = './checkpoint2/'+model


checkpoint_file = tf.train.latest_checkpoint(model_checkpoint_dir)
saver = tf.train.import_meta_graph(checkpoint_file+'.meta')


graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess,checkpoint_file)

output_node_names = ['predictions/preds']
output_graph_def = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,output_node_names=output_node_names)
#output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)
frozen_graph = 'seg_frozen2.pb'
with tf.gfile.GFile(frozen_graph,'wb') as f:
    f.write(output_graph_def.SerializeToString())
    
sess.close()
'''
graph = tf.get_default_graph()
operations = graph.get_operations()

for op in operations:
    print ('Operation:',op.name)
    #for k in op.inputs:
        print("input:",k.name,k.get_shape())
    for k in op.outputs:
        print("output:",k.name,k.get_shape())
    print('\n') 
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
for w in weights:
    shp = w.get_shape().as_list()
    print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
'''