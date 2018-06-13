# -*- coding: utf-8 -*-
"""
Created on Thu May 31 21:49:18 2018

@author: prasoon
"""
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

from tensorflow.contrib.slim.nets import resnet_v1

def resnet_v1_101_fcn(input_image, num_classes, upsample=16, is_training=True):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
          res_logits, end_points = resnet_v1.resnet_v1_101(input_image, 
                                                    num_classes, is_training=is_training,
                                                    global_pool=False,
                                                    output_stride=upsample)    
    upsample_factor = upsample
    filter_16 = tf.constant(bilinear_upsample_weights(factor=upsample_factor, number_of_classes=num_classes))
        
    l_shape = tf.shape(res_logits)
    output_shape = tf.stack([l_shape[0], upsample_factor*l_shape[1], 
                             upsample_factor*l_shape[2], l_shape[3]])
    tf_logits_4d = tf.nn.conv2d_transpose(res_logits, filter_16, output_shape, 
                                              strides=[1, upsample_factor, upsample_factor, 1])
    return tf_logits_4d
    
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    #with tf.Session() as sess:
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    graph = tf.get_default_graph()
    inp = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)


    return inp, keep, layer3, layer4, layer7

def add_fcn_old(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # call tf.stop_gradient on the 3 VGG16 layers
    #initializer=tf.truncated_normal_initializer(stddev=0.01)
    with tf.variable_scope("fcn", initializer=tf.contrib.layers.xavier_initializer(),
                                      regularizer=tf.contrib.layers.l2_regularizer(1e-3)):
        layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, padding="same", name="fcn_711")
        output_7 = tf.layers.conv2d_transpose(layer7_1x1, num_classes, kernel_size=4, strides=(2,2), padding="same", name="fcn_7out")
        vgg_layer4_out = tf.multiply(vgg_layer4_out,.01)

        layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, padding="same", name="fcn_411")

        output_47 = tf.add(output_7, layer4_1x1)        
        
        output_47 = tf.layers.conv2d_transpose(output_47, num_classes, kernel_size=4, strides=(2,2), padding="same", name="fcn_47")

        vgg_layer3_out = tf.multiply(vgg_layer3_out,.0001)

        layer3_1x1 = tf.layers.conv2d(vgg_layer3_out,num_classes,kernel_size=1, padding="same", name="fcn_311")
        
        output_473 = tf.add(layer3_1x1, output_47)

        output = tf.layers.conv2d_transpose(output_473,num_classes,kernel_size=16,strides=(8,8), padding="same", name="fcn_output")
    """with tf.name_scope("summaries_layers"):    
        tf.summary.histogram('layer3',vgg_layer3_out)
        tf.summary.histogram('layer4',vgg_layer4_out)
        tf.summary.histogram('layer7',vgg_layer7_out)
        tf.summary.histogram('output_layer',output)        
    """ 
    return output

def add_fcn(layer3, layer4, layer7, num_classes):
    filter_2 = tf.constant(bilinear_upsample_weights(factor=2, number_of_classes=num_classes))
    filter_8 = tf.constant(bilinear_upsample_weights(factor=8, number_of_classes=num_classes))
    
    with tf.variable_scope("fcn"):
            layer7_1x1 = tf.layers.conv2d(layer7, num_classes, kernel_size=1, padding="same", name="fcn_711")
    
            l_shape = tf.shape(layer7_1x1)
            print(l_shape)
            output_shape = tf.stack([l_shape[0], 2*l_shape[1], 2*l_shape[2], l_shape[3]])
            output_7 = tf.nn.conv2d_transpose(layer7_1x1, filter_2, output_shape=output_shape,strides=[1,2,2,1] )
    
            layer4 = tf.multiply(layer4,.01)
            layer4_1x1 = tf.layers.conv2d(layer4, num_classes, kernel_size=1, padding="same", name="fcn_411")
    
            output_47 = tf.add(output_7, layer4_1x1)        
    
            l_shape = tf.shape(output_47)
            output_shape = tf.stack([l_shape[0], 2*l_shape[1], 2*l_shape[2], l_shape[3]])
            output_47 = tf.nn.conv2d_transpose(output_47, filter_2, output_shape=output_shape, strides=[1,2,2,1] )
    
            layer3 = tf.multiply(layer3,.0001)
    
            layer3_1x1 = tf.layers.conv2d(layer3,num_classes,kernel_size=1, padding="same", name="fcn_311")
            
            output_473 = tf.add(layer3_1x1, output_47)
    
            l_shape = tf.shape(output_473)
            output_shape = tf.stack([l_shape[0], 8*l_shape[1], 8*l_shape[2], l_shape[3]])
            tf_output = tf.nn.conv2d_transpose(output_473, filter_8, output_shape=output_shape, strides=[1,8,8,1] )
    return tf_output


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in range(number_of_classes):        
        weights[:, :, i, i] = upsample_kernel    
    return weights