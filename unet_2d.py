from __future__ import print_function, division, absolute_import

import sklearn
import numpy as np
import tensorflow as tf
import util as U
from _weight import feedback_weight_map

layers = tf.keras.layers
initializers = tf.keras.initializers

class _Residual(tf.keras.layers.Layer):
    def __init__(self, features, name_scope):
        super(_Residual, self).__init__(name=name_scope)
        self.output_dims = features

    def build(self, input_shape):
        self.bias = self.add_weight(name='res_bias',
                            shape=(self.output_dims),
                            initializer='zeros',
                            trainable=True)
        super(_Residual, self).build(input_shape)
    
    def call(self, x1, x2, train_phase=False):
        if x1.shape[-1] < x2.shape[-1]:
            x = tf.concat([x1, tf.zeros([x1.shape[0], x1.shape[1], x1.shape[2], x2.shape[3] - x1.shape[3]])], axis=-1)
        else:
            x = x1[..., :x2.shape[-1]]
        x = x + x2
        x = x + self.bias
        return x

class _DownSev(tf.keras.Model):
    def __init__(self, features, filter_size, res, name_scope):
        super(_DownSev, self).__init__(name=name_scope)
        stddev = np.sqrt(2 / (filter_size ** 2 * features))
        self.conv1 = layers.Conv2D(features//2, 7, padding='SAME', use_bias=True,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv1')

    def call(self, input_tensor, keep_prob, train_phase):
        # conv1
        x = self.conv1(input_tensor)
        x = tf.nn.dropout(x, keep_prob)

        return x

class _DownSampling(tf.keras.Model):
    def __init__(self, features, filter_size, res, name_scope):
        super(_DownSampling, self).__init__(name=name_scope)
        stddev = np.sqrt(2 / (filter_size**2 * features))
        self.conv1 = layers.Conv2D(features//2, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv1')
        self.conv2 = layers.Conv2D(features//2, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv2')
        self.bn1 = layers.BatchNormalization(name='bn1', momentum=0.9)
        self.bn2 = layers.BatchNormalization(name='bn2', momentum=0.9)
        if res:
            self.res_block = _Residual(features//2, 'res')
        self.res = res

    def call(self, input_tensor, keep_prob, train_phase):
        # conv1

        x = input_tensor
        x = self.conv1(x)
        x = tf.nn.dropout(x, keep_prob)
        x = self.bn1(x, training=train_phase)
        x = tf.nn.relu(x)


        #
        
        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, keep_prob)
        x = self.bn2(x, training=train_phase)


        #
        if self.res:
            x = self.res_block(input_tensor, (x) , train_phase)
        x = tf.nn.relu(x)
        # res
        
        return x



class _UpSampling(tf.keras.Model):
    def __init__(self, features, filter_size, pool_size, concat_or_add, res, name_scope):
        super(_UpSampling, self).__init__(name=name_scope)
        stddev = np.sqrt(2 / (filter_size**2 * features))
        self.deconv = layers.Conv2DTranspose(features//2, filter_size, strides=(pool_size, pool_size), padding='SAME',
                                            kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                            name='deconv')
        self.bn_deconv = layers.BatchNormalization(name='bn_deconv', momentum=0.9)
        self.conv1 = layers.Conv2D(features//4, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv1')
        self.conv2 = layers.Conv2D(features//4, filter_size, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev), 
                                    name='conv2')
        self.bn1 = layers.BatchNormalization(name='bn1', momentum=0.9)
        self.bn2 = layers.BatchNormalization(name='bn2', momentum=0.9)
        
        if res:
            self.res_block = _Residual(features//4, 'res')

        self.concat_or_add = concat_or_add
        self.res = res


    def call(self, input_tensor, dw_tensor, keep_prob, train_phase):
        # deconv
        x = self.deconv(input_tensor)
        x = self.bn_deconv(x, training=train_phase)
        x = tf.nn.relu(x)
        # concatenate
        if self.concat_or_add == 'concat':
            x = self._crop_and_concat(dw_tensor, x)
        elif self.concat_or_add == 'add':
            x = self._crop_and_add(dw_tensor, x)
        else:
            raise Exception('Wrong concatenate method!')
        res_in = x
        # conv1
        x = self.conv1(x)
        x = tf.nn.dropout(x, keep_prob)
        x = self.bn1(x, training=train_phase)
        x = tf.nn.relu(x)

        #
        
        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, keep_prob)
        x = self.bn2(x, training=train_phase)


        #
        if self.res:
            x = self.res_block(res_in, (x)  , train_phase)
        x = tf.nn.relu(x)
        return x

    def _crop_and_concat(self, x1, x2):
        # x1_shape = tf.shape(x1)
        # x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        # offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        # size = [-1, x2_shape[1], x2_shape[2], -1]
        # x1_crop = tf.slice(x1, offsets, size)
        return tf.concat((x1, x2), 3)

    def _crop_and_add(self, x1, x2):
        # x1_shape = tf.shape(x1)
        # x2_shape = tf.shape(x2)
        # # offsets for the top left corner of the crop
        # offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        # size = [-1, x2_shape[1], x2_shape[2], -1]
        # x1_crop = tf.slice(x1, offsets, size)
        return x1 + x2

class Unet2D(tf.keras.Model):
    def __init__(self, n_class, n_layer, features_root, filter_size, pool_size, concat_or_add, res):
        super(Unet2D, self).__init__(name='')
        self.dw_layers = dict()
        self.up_layers = dict()
        self.max_pools = dict()
        self.dw1_layers = dict()

        for layer in range(n_layer):
            features = 2**layer*features_root
            dict_key = str(layer)
            if layer == 0:
                Dw1 = _DownSev(features, 7, res, 'dw1_%d'%layer)
                self.dw1_layers[dict_key] = Dw1
            dw = _DownSampling(features, filter_size, res, 'dw_%d'%layer)
            self.dw_layers[dict_key] = dw
            if layer < n_layer-1:
                pool = layers.MaxPool2D(pool_size=(pool_size, pool_size), padding='SAME')
                self.max_pools[dict_key] = pool

        for layer in range(n_layer-2, -1 ,-1):
            features = 2**(layer+1)*features_root
            dict_key = str(layer)
            up = _UpSampling(features, filter_size, pool_size, concat_or_add, res, 'up_%d'%layer)
            self.up_layers[dict_key] = up
        
        stddev = np.sqrt(2 / (filter_size**2 * features_root))
        self.conv_out = layers.Conv2D(n_class, 1, padding='SAME', use_bias=True,
                                    kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                    name='conv_out')


    def call(self, input_tensor, keep_prob, train_phase):
        dw_tensors = dict()
        dw1_tensors = dict()
        x = input_tensor
        dict_key = str(0)
        dw1_tensors[dict_key] = self.dw1_layers[dict_key](x, keep_prob, train_phase)
        x = dw1_tensors[dict_key]
        for i in range( len(self.dw_layers)):
            dict_key = str(i)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, keep_prob, train_phase)
            x = dw_tensors[dict_key]
            if i < len(self.max_pools):
                x = self.max_pools[dict_key](x)
        
        for i in range(len(self.up_layers) - 1, -1, -1):
            dict_key = str(i)
            x = self.up_layers[dict_key](x, dw_tensors[dict_key], keep_prob, train_phase)

        x = self.conv_out(x)
        x = tf.nn.relu(x)
        return x

class Model:
    def __init__(self, n_class, n_layer=5, features_root=16, filter_size=3, pool_size=2, weight_type=None, keep_prob=1., concat_or_add='concat', res=True):
        self.net = Unet2D(n_class, n_layer, features_root, filter_size, pool_size, concat_or_add, res)
        self.n_class = n_class
        self.weight_type = weight_type
        self.keep_prob = keep_prob

    def evaluation(self, feed_dict):
        x = tf.constant(feed_dict['x'])
        y = feed_dict['y']
        logits = self.net(x, 1., False)
        pred_prob = tf.nn.softmax(logits, axis=-1)
        loss = self.get_loss(logits, y)

        correct_pred = tf.equal(tf.argmax(pred_prob, -1), tf.argmax(y, -1))
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        pred = tf.one_hot(tf.argmax(pred_prob, -1), self.n_class)
        flat_labels = tf.reshape(y, [-1, self.n_class])
        flat_pred = tf.reshape(pred, [-1, self.n_class])
        flat_pred_prob = tf.reshape(pred_prob, [-1, self.n_class])

        # dice
        eps = 1e-5
        intersection = tf.reduce_sum(flat_pred * flat_labels, axis=0)
        sum_ = eps + tf.reduce_sum(flat_pred + flat_labels, axis=0)
        dice = 2 * intersection / sum_

        # iou
        iou = intersection / (sum_ - intersection)

        result = {'prediction':np.array(pred_prob),
                  'loss':np.array(loss),
                  'acc':np.array(acc),
                  'dice':np.array(dice),
                  'iou':np.array(iou)}
        return result

    def evaluation_dice(self, feed_dict):
        x = tf.constant(feed_dict['x'])
        y = feed_dict['y']
        logits = self.net(x, 1., False)
        pred_prob = tf.nn.softmax(logits, axis=-1)
        pred = tf.one_hot(tf.argmax(pred_prob, -1), self.n_class)
        flat_labels = tf.reshape(y, [-1, self.n_class])
        flat_pred = tf.reshape(pred, [-1, self.n_class])

        # dice
        eps = 1e-5
        intersection = tf.reduce_sum(flat_pred * flat_labels, axis=0)
        sum_ = eps + tf.reduce_sum(flat_pred + flat_labels, axis=0)
        dice = 2 * intersection / sum_

        # # iou
        # iou = intersection / (sum_ - intersection)

        return np.array(dice) #, iou


    def get_grads(self, feed_dict): 
        x = tf.constant(feed_dict['x'])
        y = feed_dict['y']
        with tf.GradientTape() as grads_tape:
            logits = self.net(x, self.keep_prob, True)
            loss = self.get_loss(logits, y)
        grads = grads_tape.gradient(loss, self.net.variables)

        #grads = tf.gradients(loss, self.net.variables)
        return [grad if grad is not None else tf.zeros_like(var)
                for var, grad in zip(self.net.variables, grads)]

    def get_logits(self, x): 
        logits = self.net(x, self.keep_prob, True)

        return logits

    def predict(self, x):
        x = tf.constant(x)
        return tf.nn.softmax(self.net(x, 1., False), axis=-1)

    def get_loss(self, logits, labels):
        flat_logits = tf.reshape(logits, [-1, self.n_class])
        flat_labels = tf.reshape(labels, [-1, self.n_class])
        loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)
        if self.weight_type is None:
            loss = tf.reduce_mean(loss_map)
        else:
            probs = tf.nn.softmax(logits, axis=-1)
            flat_probs = tf.reshape(probs, [-1, self.n_class])
			if self.weight_type == 'feedback':
                weight_map = feedback_weight_map(flat_probs, flat_labels, 3, 100)
            else:
                raise ValueError("Unknown weight type: "%self.weight_type)

            loss = tf.reduce_mean(tf.multiply(loss_map, weight_map)) 

        return loss




    
        
        


        
        

