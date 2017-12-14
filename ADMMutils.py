#!/usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import _pickle as pickle
import argparse
import os
import re
import sys
import tarfile
import numpy as np
from six.moves import urllib
from scipy.misc import imread
from scipy.ndimage import zoom
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
def frobenius_norm_square(tensor):
    squareroot_tensor = tf.square(tensor)
    frobenius_norm2 = tf.reduce_sum(squareroot_tensor)
    return frobenius_norm2
    
def frobenius_norm(tensor):
    frobenius_norm = tf.sqrt(frobenius_norm_square(tensor))
    return frobenius_norm

def frobenius_norm_block(tensor,dim):
    squareroot_tensor = tf.square(tensor)
    tensor_sum = tf.reduce_sum(squareroot_tensor,dim)
    frobenius_norm = tf.sqrt(tensor_sum)
    return frobenius_norm

def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    dest_directory = '.data/'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def ImageProducer(filename_queue):
    label_bytes = 1; 
    height = 32; width = 32; depth = 3
    image_bytes = height * width * depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    rec = tf.decode_raw(value, tf.uint8)
    label = tf.cast(tf.strided_slice(rec, [0], [label_bytes]), tf.int32)
    label = tf.reshape(label,[1])
    #image = tf.slice(rec, [label_bytes], [image_bytes])#tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),[depth,height,width])
    image = tf.strided_slice(rec, [label_bytes], [label_bytes + image_bytes])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image,[1,image_bytes])  
    image = tf.subtract(image,tf.reduce_mean(image))
    scale = tf.constant(55.); thresh = tf.constant(1.)
    std_val  = tf.div(tf.sqrt(tf.reduce_sum(tf.square(image))),scale); 
    f4 = lambda: std_val
    f5 = lambda: thresh
    normalizer = tf.cond(tf.less(std_val,1e-8),f5,f4)
    image = tf.div(image,normalizer)
    image = tf.subtract(image,tf.reduce_mean(image))
    depth_major = tf.reshape(image,[depth,height,width])
    image = tf.transpose(depth_major, [1, 2, 0])
    return image, label
       
def conv(inputx, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding) 
    if group==1:
        conv = convolve(inputx, kernel)
    else:
        input_groups = tf.split(3, group, inputx)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())    
    #return  tf.nn.bias_add(conv, biases) 
 #####################################################   
def block_shrinkage_conv(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V); one_val = tf.constant(1.0)
    print(V_shape)
    b = tf.div(mu,rho)
    V_shape1 = tf.concat([tf.multiply(tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1])),tf.multiply(tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1]))], 0)
    V = tf.reshape(tf.transpose(V,perm=[2,3,0,1]),V_shape1)
    norm_V = frobenius_norm_block(V,1)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape1,[1],[1]),'float'))
    zero_part = tf.zeros(V_shape1)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
#    f4 = lambda: tf.greater_equal(tf.truediv(tf.add(tf.reduce_min(fro),tf.reduce_mean(fro)),2.0),fro)
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.multiply(coef,tf.cast(V_shape1[0],'float'))),f4,f5)
    G = tf.where(zero_ind,zero_part,tf.multiply(tf.subtract(one_val,tf.div(b,tf.reshape(norm_V,[-1,1]))),V))
    G_shape = tf.concat([tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1]),tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1])],0)
    G = tf.transpose(tf.reshape(G,G_shape),perm=[2,3,0,1])
    return G,zero_ind
    
def block_shrinkage_fc(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V); one_val = tf.constant(1.0) 
    b = tf.div(mu,rho)
    norm_V = frobenius_norm_block(V,0)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape,[0],[1]),'float'))
    zero_part = tf.zeros(V_shape)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.multiply(coef,tf.reshape(tf.cast(tf.slice(V_shape,[1],[1]),'float'),[]))),f4,f5)
    G = tf.transpose(tf.select(zero_ind,tf.transpose(zero_part),tf.transpose(tf.multiply(V,tf.transpose(tf.subtract(one_val,tf.div(b,tf.reshape(norm_V,[-1,1]))))))))
    return G,zero_ind
def block_truncate_conv(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V) 
    b = tf.sqrt(tf.div(tf.multiply(2.,mu),rho)) #threshold
    # Reshape the 4D tensor of weights to a 2D matrix with rows containing the conv filters in vectorized form.
    V_shape1 = tf.concat(0,[tf.multiply(tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1])),tf.multiply(tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1]))])
    V = tf.reshape(tf.transpose(V,perm=[2,3,0,1]),V_shape1)
    norm_V = frobenius_norm_block(V,1)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape1,[1],[1]),'float'))
    # Implementation of Eq.10 in the paper using if condition inside the TensorFlow graph with tf.cond
    zero_part = tf.zeros(V_shape1)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
    # You can pass parameters to the functions in tf.cond() using lambda
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.multiply(coef,tf.cast(V_shape1[0],'float'))),f4,f5)
    G = tf.select(zero_ind,zero_part,V) 
    G_shape = tf.concat(0,[tf.slice(V_shape,[2],[1]),tf.slice(V_shape,[3],[1]),tf.slice(V_shape,[0],[1]),tf.slice(V_shape,[1],[1])])
    G = tf.transpose(tf.reshape(G,G_shape),perm=[2,3,0,1])
    return G,zero_ind
    
def block_truncate_fc(V,mu,rho):
    coef = 0.5
    V_shape = tf.shape(V) 
    b = tf.sqrt(tf.div(tf.multiply(2.,mu),rho)) #threshold
    norm_V = frobenius_norm_block(V,0)  
    norm_V_per_dimension = tf.div(norm_V,tf.cast(tf.slice(V_shape,[0],[1]),'float'))
    zero_part = tf.zeros(V_shape)
    zero_ind = tf.greater_equal(b,norm_V_per_dimension)
    num_zero = tf.reduce_sum(tf.cast(zero_ind,'float'))
    f4 = lambda: tf.greater_equal(tf.reduce_mean(norm_V),norm_V)
    f5 = lambda: zero_ind
    zero_ind = tf.cond(tf.greater(num_zero,tf.multiply(coef,tf.reshape(tf.cast(tf.slice(V_shape,[1],[1]),'float'),[]))),f4,f5)
    G = tf.transpose(tf.select(zero_ind,tf.transpose(zero_part),tf.transpose(V))) 
    return G,zero_ind    