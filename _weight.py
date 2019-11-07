import numpy as np
import tensorflow as tf

def feedback_weight_map(flat_probs, flat_labels, beta, op):
    '''
    return the feedback weight map in 1-D tensor
    :param flat_probs: prediction tensor in shape [-1, n_class]
    :param flat_labels: ground truth tensor in shape [-1, n_class]
    '''
    probs = tf.reduce_sum(flat_probs*flat_labels, axis=-1)
    weight_map = tf.exp(-tf.pow(probs, beta)*tf.log(tf.constant(op, "float")))   
    return weight_map 
