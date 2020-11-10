# Copyright 2020 Hassan Akbari, Columbia University. All Rights Reserved.
# Email: ha2436@columbia.edu
# Website: hassanakbari.com
# GitHub: https://github.com/hassanhub
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""
This is a Tensorflow implementation of the original SlowFast network as described in [1].
This code also includes a 3D version of the original ResNet (both v1 [2], and v2 [3]).
NL blocks (as described in [1]) are also used and the implementation matches with [4].

[1] Feichtenhofer, Christoph, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "Slowfast networks for video recognition."
    In Proceedings of the IEEE International Conference on Computer Vision, pp. 6202-6211. 2019.
[2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    "Deep residual learning for image recognition."
    In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
[3] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
    "Identity mappings in deep residual networks."
    In European conference on computer vision, pp. 630-645. Springer, Cham, 2016.
[4] Wang, Xiaolong, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7794-7803. 2018.

Compatibility: Tensorflow v2.2
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dense
from tensorflow.keras.layers import MaxPool1D, MaxPool2D, MaxPool3D
from tensorflow.keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import Concatenate, BatchNormalization, Dropout, Add
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model
import numpy as np

def auto_pad(inputs,
          kernel_size,
          data_format):
    
    """
    This function replaces the padding implementation in original tensorflow.
    It also avoids negative dimension by automatically padding given the input kernel size (for each dimension).
    """

    islist = isinstance(kernel_size, list)

    kernel_size = np.array(kernel_size)
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if islist:
        paddings = np.concatenate([pad_beg[:,np.newaxis], pad_end[:,np.newaxis]],axis=1)
        paddings = [list(p) for p in paddings]
    else:
        paddings = [[pad_beg,pad_end]]*3

    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0],
                                         [0, 0]]+paddings)
    else:
        padded_inputs = tf.pad(tensor=inputs,
                               paddings=[[0, 0]]+paddings+[[0, 0]])
    return padded_inputs

class Permute(Layer):
    def __init__(self,
                 perm_dims,
                 name='Permute',
                 **kwargs):
        super(Permute,self).__init__(name=name,**kwargs)
        self.perm_dims = perm_dims

    def call(self,
             inputs,
             training=None):
        return tf.transpose(a=inputs, perm=self.perm_dims)

class Reduce_Mean(Layer):
    def __init__(self,
                 reduce_dims,
                 name='Reduce_Mean',
                 **kwargs):
        super(Reduce_Mean,self).__init__(name=name,**kwargs)
        self.reduce_dims = reduce_dims

    def call(self,
             inputs,
             training=None):
        return tf.reduce_mean(inputs, self.reduce_dims)

class ReLU(Layer):
    def __init__(self,
                 name='ReLU',
                 **kwargs):
        super(ReLU,self).__init__(name=name, **kwargs)

    def call(self,
             inputs,
             training=None):
        return tf.nn.relu(inputs)


class Concat(Concatenate):
    def __init__(self, 
               data_format,
               name='Concat',
               **kwargs):

        axis=1 if data_format == 'channels_first' else -1
        super(Concat, self).__init__(axis=axis,
                                        name=name,
                                        **kwargs)

class BatchNorm(BatchNormalization):
    def __init__(self, 
               data_format,
               name='BatchNorm',
               **kwargs):

        axis=1 if data_format == 'channels_first' else -1
        super(BatchNorm, self).__init__(axis=axis,
                                        center=True,
                                        scale=True,
                                        name=name,
                                        **kwargs)

class ConvXD(Layer):
    def __init__(self,
               filters, 
               kernel_size, 
               strides, 
               data_format,
               name,
               use_bias=False,
               **kwargs):
        super(ConvXD,self).__init__(name=name,**kwargs)
        #self.name = name
        self.pad = False
        if isinstance(strides,list):
            self.pad = any(np.array(strides) > 1)
        else:
            self.pad = strides>1

        if self.pad:
            padding = 'valid'
        else:
            padding = 'same'

        self.conv = Conv3D(filters=filters, 
                            kernel_size=kernel_size, 
                            strides=strides,
                            padding=padding, 
                            use_bias=use_bias,
                            data_format=data_format,
                            name='conv_3d')

        self.data_format = data_format
        self.kernel_size = kernel_size


    def call(self,
           inputs,
           training=None):
        if self.pad:
            inputs = auto_pad(inputs = inputs, 
                            kernel_size = self.kernel_size, 
                            data_format = self.data_format)
            
        outputs = self.conv(inputs)
        return outputs

class FastToSlow(Layer):
    def __init__(self,
                 target_filers,
                 fusion_kernel,
                 alpha,
                 use_bias,
                 fuse_method,
                 pre_activation,
                 momentum,
                 epsilon,
                 data_format,
                 name='Fuse',
                 **kwargs):
        super(FastToSlow,self).__init__(name=name,**kwargs)
        self.pre_activation = pre_activation

        self.conv_f2s = ConvXD(filters=target_filers, 
                               kernel_size=[fusion_kernel,1,1], 
                               strides=[alpha,1,1], 
                               data_format=data_format,
                               use_bias=use_bias,
                               name='Conv_Fuse')
        if not self.pre_activation:
            self.bn = BatchNorm(data_format=data_format,
                                momentum=momentum,
                                epsilon=epsilon,
                                name='BatchNorm_Fuse')

        if fuse_method=='concat':
            self.fuse = Concat(data_format=data_format,
                                name='Concat_Fuse')
        elif fuse_method=='add':
            self.fuse = Add(name='Add_Fuse')

    def call(self,
            inputs_slow,
            inputs_fast,
            training):
        inputs_fast = self.conv_f2s(inputs_fast)
        if not self.pre_activation:
            inputs_fast = self.bn(inputs_fast,training)
            inputs_fast = tf.nn.relu(inputs_fast)
        outputs = self.fuse([inputs_slow, inputs_fast])

        return outputs


class BasicResConvs(Layer):
    def __init__(self,
                filters, 
                strides,
                kernel_sizes,
                shortcut,
                pre_activation,
                momentum,
                epsilon,
                data_format,
                name,
                **kwargs):

        super(BasicResConvs,self).__init__(name=name,**kwargs)

        self.bn_a = BatchNorm(data_format=data_format,
                              momentum=momentum,
                              epsilon=epsilon,
                              name='BatchNorm_a')
        
        self.conv_a = ConvXD(filters=filters,
                               kernel_size=kernel_sizes[0],
                               strides=strides,
                               data_format=data_format,
                               name='Conv_a')
        
        self.bn_b = BatchNorm(data_format=data_format,
                              momentum=momentum,
                              epsilon=epsilon,
                              gamma_initializer='zeros',
                              name='BatchNorm_b')
        
        self.conv_b = ConvXD(filters=filters,
                               kernel_size=kernel_sizes[1],
                               strides=1,
                               data_format=data_format,
                               name='Conv_b')

        if shortcut=='identity':
            self.res_op = tf.identity
        
        elif shortcut=='projection':
            self.res_op = ConvXD(filters=filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    name='Shortcut/Conv_s')
            if not pre_activation:
                self.bn_r = BatchNorm(data_format=data_format,
                                  momentum=momentum,
                                  epsilon=epsilon,
                                  name='Shortcut/BatchNorm_s')

        else:
            raise ValueError('Shortcut type not supported.')

class BottleneckResConvs(Layer):
    def __init__(self,
                filters, 
                strides,
                kernel_sizes,
                shortcut,
                pre_activation,
                momentum,
                epsilon,
                data_format,
                name,
                **kwargs):
        super(BottleneckResConvs,self).__init__(name=name,**kwargs)

        self.bn_a = BatchNorm(data_format=data_format,
                              momentum=momentum,
                              epsilon=epsilon,
                              name='BatchNorm_a')
        
        self.conv_a = ConvXD(filters=filters,
                               kernel_size=kernel_sizes[0],
                               strides=1,
                               data_format=data_format,
                               name='Conv_a')
        
        self.bn_b = BatchNorm(data_format=data_format,
                              momentum=momentum,
                              epsilon=epsilon,
                              name='BatchNorm_b')
        
        self.conv_b = ConvXD(filters=filters,
                               kernel_size=kernel_sizes[1],
                               strides=strides,
                               data_format=data_format,
                               name='Conv_b')

        self.bn_c = BatchNorm(data_format=data_format,
                              momentum=momentum,
                              epsilon=epsilon,
                              gamma_initializer='zeros',
                              name='BatchNorm_c')
        
        self.conv_c = ConvXD(filters=4 * filters,
                               kernel_size=kernel_sizes[2],
                               strides=1,
                               data_format=data_format,
                               name='Conv_c')

        if shortcut=='identity':
            self.res_op = tf.identity
        
        elif shortcut=='projection':
            self.res_op = ConvXD(filters=4 * filters,
                                    kernel_size=1,
                                    strides=strides,
                                    data_format=data_format,
                                    name='Shortcut/Conv_s')
            if not pre_activation:
                self.bn_r = BatchNorm(data_format=data_format,
                                  momentum=momentum,
                                  epsilon=epsilon,
                                  name='Shortcut/BatchNorm_s')
        else:
            raise ValueError('Shortcut type not supported.')

class BasicResLayer(BasicResConvs):
    """
        The basic (building) residual block as proposed in [1], [2].
        

    """
    def __init__(self,
                filters, 
                strides,
                kernel_sizes,
                shortcut='identity',
                pre_activation=True,
                momentum=0.9,
                epsilon=1e-5,
                data_format='channels_last',
                name='Res_Layer',
                **kwargs):

        super(BasicResLayer,self).__init__(filters=filters, 
                                            strides=strides,
                                            kernel_sizes=kernel_sizes,
                                            shortcut=shortcut,
                                            pre_activation=pre_activation,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            data_format=data_format,
                                            name=name,
                                            **kwargs)

        
        self.shortcut = shortcut
        self.pre_activation = True

    def call(self,
            inputs,
            training=None):

        if self.pre_activation:
            return self._call_v2(inputs, training)
        else:
            return self._call_v1(inputs, training)

    def _call_v1(self,
               inputs,
               training):

        res_out = self.res_op(inputs)
        if self.shortcut == 'projection':
            res_out = self.bn_r(res_out, training=training)

        inputs = self.conv_a(inputs)
        inputs = self.bn_a(inputs, training=training)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv_b(inputs)
        inputs = self.bn_b(inputs, training=training)

        inputs += res_out
        outputs = tf.nn.relu(inputs)

        return outputs

    def _call_v2(self,
               inputs,
               training):

        if self.shortcut == 'identity':
            res_out = self.res_op(inputs)

        inputs = self.bn_a(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        
        if self.shortcut == 'projection':
            res_out = self.res_op(inputs)

        inputs = self.conv_a(inputs)
        
        inputs = self.bn_b(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv_b(inputs)
        
        outputs = inputs + res_out
        
        return outputs

class BottleneckResLayer(BottleneckResConvs):
    """
        The bottleneck residual block as proposed in [1], [2].
        

    """
    def __init__(self,
                filters,
                strides,
                kernel_sizes,
                shortcut='identity',
                pre_activation=True,
                momentum=0.9,
                epsilon=1e-5,
                data_format='channels_last',
                name='Res_Layer',
                **kwargs):
        super(BottleneckResLayer, self).__init__(filters=filters,
                                                strides=strides,
                                                kernel_sizes=kernel_sizes,
                                                shortcut=shortcut,
                                                pre_activation=pre_activation,
                                                momentum=momentum,
                                                epsilon=epsilon,
                                                data_format=data_format,
                                                name=name,
                                                **kwargs)

        
        self.shortcut = shortcut
        self.pre_activation = pre_activation

    def call(self,
            inputs,
            training):

        if self.pre_activation:
            return self._call_v2(inputs, training)
        else:
            return self._call_v1(inputs, training)

    def _call_v1(self,
               inputs,
               training):

        res_out = self.res_op(inputs)
        if self.shortcut == 'projection':
            res_out = self.bn_r(res_out, training=training)

        inputs = self.conv_a(inputs)
        inputs = self.bn_a(inputs, training=training)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv_b(inputs)
        inputs = self.bn_b(inputs, training=training)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv_c(inputs)
        inputs = self.bn_c(inputs, training=training)

        inputs += res_out
        outputs = tf.nn.relu(inputs)

        return outputs

    def _call_v2(self,
               inputs,
               training):

        if self.shortcut == 'identity':
            res_out = self.res_op(inputs)

        inputs = self.bn_a(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        
        if self.shortcut == 'projection':
            res_out = self.res_op(inputs)

        inputs = self.conv_a(inputs)
        
        inputs = self.bn_b(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv_b(inputs)
        
        inputs = self.bn_c(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv_c(inputs)

        outputs = inputs + res_out
        
        return outputs

class ResidualBlock(Layer):
    def __init__(self,
                 filters,
                 kernel_sizes,
                 strides,
                 num_layers,
                 res_layer,
                 momentum,
                 epsilon,
                 pre_activation=True,
                 data_format='channels_last',
                 name='Res_Block',
                 **kwargs):

        super(ResidualBlock,self).__init__(name=name,**kwargs)
        #only the first layer uses projection shortcut
        shortcuts = ['projection']+['identity']*(num_layers-1)

        #only the first layer uses strides>1
        strides = [strides]+[1]*(num_layers-1)

        #building the resnet model
        self.res_layers = []
        for n in range(num_layers):
            self.res_layers.append(res_layer(filters=filters,
                                             kernel_sizes=kernel_sizes,
                                             strides=strides[n],
                                             shortcut=shortcuts[n],
                                             momentum=momentum,
                                             epsilon=epsilon,
                                             pre_activation=pre_activation,
                                             data_format=data_format,
                                             name='Res_Layer_{}'.format(n),
                                             **kwargs))
    
    def call(self,
             inputs,
             training=None):
        for layer in self.res_layers:
            inputs = layer(inputs=inputs, training=training)
        return inputs


class ResNet(Layer):
    def __init__(self,
                 num_classes=1000,
                 first_conv_filters=64,
                 first_conv_size=7,
                 first_conv_strides=2,
                 first_pool_size=3,
                 first_pool_strides=2,
                 res_layer=BottleneckResLayer,
                 block_sizes=[3, 4, 23, 3],
                 block_kernels=[[1,3,1], [1,3,1], [1,3,1], [1,3,1]],
                 block_strides=[1, 2, 2, 2],
                 temporal_sampling_rate=4,
                 spatial_sampling_rate=1,
                 pre_activation=True,
                 momentum=0.9,
                 epsilon=1e-5,
                 data_format='channels_last',
                 name='ResNet',
                 **kwargs):

        #default values are for a 2D ResNet101 proper for 224x224 input size
        super(ResNet, self).__init__(name=name, **kwargs)

        self.res_layer = res_layer
        self.pre_activation = pre_activation
        self.data_format = data_format

        self.pre_block = []
        
        #reshape inputs if channels are first
        if self.data_format == 'channels_first':
            self.pre_block.append(Permute(perm_dims=[0, 4, 1, 2, 3],
                                          name='Pre_Block/Permute_p'))

        #first convolution
        self.pre_block.append(ConvXD(filters=first_conv_filters,
                                     kernel_size=first_conv_size,
                                     strides=first_conv_strides,
                                     data_format=data_format,
                                     name='Pre_Block/Conv_p'))

        #if there is no pre_activation, a BN+ReLU should be applied before feeding
        #to pooling layer and residual blocks
        if not self.pre_activation:
            self.pre_block.append(BatchNorm(data_format=data_format,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name='Pre_Block/BatchNorm_p'))
            self.pre_block.append(ReLU())

        #pooling before residual blocks
        self.pre_block.append(MaxPool3D(pool_size=first_pool_size, 
                                        strides=first_pool_strides, 
                                        padding='same', 
                                        data_format=data_format,
                                        name='Pre_Block/Pool_p'))

        #main residual blocks
        self.residual_blocks = []
        for n,num_layers in enumerate(block_sizes):
            filters = first_conv_filters * (2 ** n)
            kernel_sizes = block_kernels[n]
            strides = block_strides[n]
            self.residual_blocks.append(ResidualBlock(filters=filters,
                                                     kernel_sizes=kernel_sizes,
                                                     strides=strides,
                                                     num_layers=num_layers,
                                                     res_layer=res_layer,
                                                     pre_activation=pre_activation,
                                                     momentum=momentum,
                                                     epsilon=epsilon,
                                                     data_format=data_format,
                                                     name='Res_Block_{}'.format(n+2)))

        #apply final BN+ReLU if pre_activation == True
        self.post_block = []
        if self.pre_activation:
            self.post_block.append(BatchNorm(data_format=data_format,
                                             momentum=momentum,
                                             epsilon=epsilon,
                                             name='Post_Block/BatchNorm_p'))
            self.post_block.append(ReLU())

        #reshape outputs before return if channels are first
        if self.data_format == 'channels_first':
            self.post_block.append(Permute(perm_dims=[0,2,3,4,1],
                                           name='Post_Block/Permute_p'))
        
        self.spatial_pool = Reduce_Mean(reduce_dims=[2,3],
                                       name='Spatial_Pool')
        self.global_pool = Reduce_Mean(reduce_dims=[1,2,3],
                                       name='Global_Pool')
        
        self.temporal_sample = AveragePooling3D(pool_size=[temporal_sampling_rate,1,1],
                                                strides=[temporal_sampling_rate,1,1],
                                                padding='same',
                                                data_format=data_format,
                                                name='Temporal_Sample')

        self.spatial_sample = AveragePooling3D(pool_size=[1,spatial_sampling_rate,spatial_sampling_rate],
                                                strides=[1,1,1],
                                                padding='valid',
                                                data_format=data_format,
                                                name='Spatial_Sample')

        if num_classes is None:
            self.classify = tf.identity
        else:
            self.classify = Dense(num_classes, name='Classification')

    #@tf.function
    def call(self,
             inputs,
             training):

        with tf.name_scope('ResNet'):
            for layer in self.pre_block:
                inputs = layer(inputs=inputs, training=training)

            for layer in self.residual_blocks:
                inputs = layer(inputs=inputs, training=training)

            for layer in self.post_block:
                inputs = layer(inputs=inputs, training=training)

            features = inputs
            features_pooled = self.global_pool(features)
            logits = self.classify(features_pooled)

        return features, features_pooled, logits

class SlowFast(Model):
    def __init__(self,
                 num_classes,
                 slow_config,
                 fast_config,
                 pre_activation,
                 temporal_sampling_rate,
                 spatial_sampling_rate,
                 fusion_kernel,
                 alpha,
                 beta,
                 tau,
                 dropout_rate,
                 epsilon,
                 momentum,
                 data_format,
                 name='SlowFast',
                 **kwargs):
        super(SlowFast,self).__init__(name=name, **kwargs)

        #DEFAUL VALUES NOT SPECIFIED IN CONFIGS
        first_conv_filters = 64
        res_layer = BottleneckResLayer
        pre_activation = False

        self.backbone_freezed = False
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.Slow = ResNet(num_classes=None,
                            first_conv_filters=first_conv_filters,
                            first_conv_size=slow_config['CONV1_SIZE'],
                            first_conv_strides=slow_config['CONV1_STRIDES'],
                            first_pool_size=slow_config['POOL1_SIZE'],
                            first_pool_strides=slow_config['POOL1_STRIDES'],
                            block_sizes=slow_config['BLOCK_SIZES'],
                            block_kernels=slow_config['BLOCK_KERNELS'],
                            block_strides=slow_config['BLOCK_STRIDES'],
                            temporal_sampling_rate=temporal_sampling_rate,
                            spatial_sampling_rate=spatial_sampling_rate,
                            res_layer=res_layer,
                            pre_activation=pre_activation,
                            momentum=epsilon,
                            epsilon=momentum,
                            data_format=data_format,
                            name='Slow')

        self.Fast = ResNet(num_classes=None,
                            first_conv_filters=int(first_conv_filters * beta),
                            first_conv_size=fast_config['CONV1_SIZE'],
                            first_conv_strides=fast_config['CONV1_STRIDES'],
                            first_pool_size=fast_config['POOL1_SIZE'],
                            first_pool_strides=fast_config['POOL1_STRIDES'],
                            block_sizes=fast_config['BLOCK_SIZES'],
                            block_kernels=fast_config['BLOCK_KERNELS'],
                            block_strides=fast_config['BLOCK_STRIDES'],
                            temporal_sampling_rate=temporal_sampling_rate * tau,
                            spatial_sampling_rate=spatial_sampling_rate,
                            res_layer=res_layer,
                            pre_activation=pre_activation,
                            momentum=epsilon,
                            epsilon=momentum,
                            data_format=data_format,
                            name='Fast')

        if self.num_classes is not None:
            self.dropout = Dropout(dropout_rate, name='Dropout')
            self.classify = Dense(num_classes, name='Classification')

        self.res_blocks = ['res2', 'res3', 'res4', 'res5']
        self.fuse_points = ['pool1', 'res2', 'res3', 'res4']
        self.fuse_layers = []
        for n,fuse_point in enumerate(self.fuse_points):
            if fuse_point == 'pool1':
                f2s_filters = 2 * int(beta * first_conv_filters) #2*beta*C
            else:
                f2s_filters = 2 * int(beta * first_conv_filters * (2 ** (n-1))) #2*beta*C
                if res_layer == BottleneckResLayer:
                    f2s_filters *= 4 #output of each block has 4x filters than first layer (only in BottleneckResLayer case)
            self.fuse_layers.append(FastToSlow(target_filers=f2s_filters,
                                         fusion_kernel=fusion_kernel,
                                         alpha=alpha,
                                         use_bias=False,
                                         fuse_method='concat',
                                         pre_activation=False,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         data_format=data_format,
                                         name='Fuse/Fuse_{}'.format(fuse_point)))
    
    def freeze_backbone(self):
        self.Slow.trainable = False
        self.Fast.trainable = False
        for layer in self.fuse_layers:
            layer.trainable = False

        self.backbone_freezed = True

    def _img_preprocess(self,frame):
        frame = tf.cast(frame, dtype=tf.float32)
        frame = tf.divide(frame, 255.)
        frame = tf.subtract(frame, .5)
        frame = tf.multiply(frame, 2.0)
        return frame

    def set_strategy(self,
                     strategy):
        self.strategy = strategy

    @tf.function
    def _init_0(self,
             inputs):
        self.call(inputs=inputs,training=False)
    
    @tf.function
    def _init_d(self,
                inputs):
        self.strategy.run(self._init_0,args=(inputs,))

    def init(self,
             input_shape):
        inputs = tf.random.uniform(input_shape, minval=0, maxval=255, dtype=tf.int32)
        self._init_0(tf.cast(inputs,dtype=tf.uint8))

    def distributed_init(self,
                         input_shape):
        inputs = tf.random.uniform(input_shape, minval=0, maxval=255, dtype=tf.int32)
        self._init_d(tf.cast(inputs,dtype=tf.uint8))
        

    @tf.function
    def call(self,
             inputs,
             training):

        inputs = self._img_preprocess(inputs)
        inputs_slow = inputs[:,::self.tau,:]
        inputs_fast = inputs[:,::int(self.tau/self.alpha),:]
        inputs_f2s = {}

        backbone_training = training and not self.backbone_freezed

        with tf.name_scope('SlowFast'):
            with tf.name_scope('Fast'):
                #pre blocks
                for layer in self.Fast.pre_block:
                    inputs_fast = layer(inputs=inputs_fast, training=backbone_training)

                #residual blocks
                for n,res_block in enumerate(self.res_blocks):
                    inputs_f2s[res_block] = inputs_fast
                    inputs_fast = self.Fast.residual_blocks[n](inputs=inputs_fast, training=backbone_training)

                #post blocks
                for layer in self.Fast.post_block:
                    inputs_fast = layer(inputs=inputs_fast, training=backbone_training)

            
            with tf.name_scope('Slow'):
                #pre blocks
                for layer in self.Slow.pre_block:
                    inputs_slow = layer(inputs=inputs_slow, training=backbone_training)
                
                #residual blocks
                for n,res_block in enumerate(self.res_blocks):
                    inputs_slow = self.fuse_layers[n](inputs_slow=inputs_slow,
                                                      inputs_fast=inputs_f2s[res_block],
                                                      training=backbone_training)
                    inputs_slow = self.Slow.residual_blocks[n](inputs=inputs_slow, training=backbone_training)

                #post blocks
                for layer in self.Slow.post_block:
                    inputs_slow = layer(inputs=inputs_slow, training=backbone_training)

            features_slow = inputs_slow
            features_fast = inputs_fast

            features_slow = self.Slow.temporal_sample(features_slow)
            features_slow = self.Slow.spatial_sample(features_slow)
            features_fast = self.Fast.temporal_sample(features_fast)
            features_fast = self.Fast.spatial_sample(features_fast)

            if self.num_classes is None:
                features_pooled = tf.concat([features_slow, features_fast], axis=-1)
                return features_pooled
            else:
                features_slow_pooled = self.Slow.global_pool(features_slow)
                features_fast_pooled = self.Fast.global_pool(features_fast)
                features_pooled = tf.concat([features_slow_pooled, features_fast_pooled], axis=1)
                features_pooled = self.dropout(features_pooled, training)
                predictions = tf.nn.softmax(self.classify(features_pooled))

                return predictions