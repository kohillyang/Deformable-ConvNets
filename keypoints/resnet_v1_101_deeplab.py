# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# --------------------------------------------------------

import mxnet as mx

class resnet_v1_101_deeplab():
    def Conv(self,data,num_filter = 512,pad = (1,1),kernel = (3,3),no_bias = True):
        conv = mx.sym.Convolution(data = data,num_filter=num_filter,pad=pad,kernel=kernel)
        bn = mx.sym.BatchNorm(data = conv,use_global_stats=True,fix_gamma=False,eps = self.eps)
        relu = mx.sym.Activation(data = bn,act_type='relu')
        return  relu
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 4096
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_conv(self, data):
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False,
                                       eps=self.eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        #         pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
        #                                   stride=(2, 2), pool_type='max')
        pool1 = conv1_relu
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')
        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=self.eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')
        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=self.eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')

        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=self.eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b22_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512,
                                               pad=(2, 2), kernel=(3, 3), dilate=(2, 2), stride=(1, 1), no_bias=True)
        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512,
                                               pad=(2, 2), kernel=(3, 3), dilate=(2, 2), stride=(1, 1), no_bias=True)
        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512,
                                               pad=(2, 2), kernel=(3, 3), dilate=(2, 2), stride=(1, 1), no_bias=True)
        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')
        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048,
                                               pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=self.eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')
        return res5c_relu

    def get_train_symbol(self, num_classes):
        """
        get symbol for training
        :param num_classes: num of classes
        :return: the symbol for training
        """
        data = mx.symbol.Variable(name="data")

        seg_cls_gt = mx.symbol.Variable(name='label')

        # shared convolutional layers
        conv_feat = self.get_resnet_conv(data)

        # subsequent fc layers by haozhi
        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6",
                                    bias=fc6_bias, weight=fc6_weight, workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score",
                                      bias=score_bias, weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16),
                                             num_group=num_classes, no_bias=True, name='upsampling',
                                             attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(8, 8), name='croped_score')
        softmax = mx.symbol.SoftmaxOutput(data=croped_score, label=seg_cls_gt, normalization='valid', multi_output=True,
                                          use_ignore=True, ignore_label=255, name="softmax")

        return softmax

    def get_test_symbol(self, num_classes):
        """
        get symbol for testing
        :param num_classes: num of classes
        :return: the symbol for testing
        """
        data = mx.symbol.Variable(name="data")

        # shared convolutional layers
        conv_feat = self.get_resnet_conv(data)

        fc6_bias = mx.symbol.Variable('fc6_bias', lr_mult=2.0)
        fc6_weight = mx.symbol.Variable('fc6_weight', lr_mult=1.0)

        fc6 = mx.symbol.Convolution(
            data=conv_feat, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc6", bias=fc6_bias, weight=fc6_weight,
            workspace=self.workspace)
        relu_fc6 = mx.sym.Activation(data=fc6, act_type='relu', name='relu_fc6')

        score_bias = mx.symbol.Variable('score_bias', lr_mult=2.0)
        score_weight = mx.symbol.Variable('score_weight', lr_mult=1.0)

        score = mx.symbol.Convolution(
            data=relu_fc6, kernel=(1, 1), pad=(0, 0), num_filter=num_classes, name="score", bias=score_bias,
            weight=score_weight, workspace=self.workspace)

        upsampling = mx.symbol.Deconvolution(
            data=score, num_filter=num_classes, kernel=(32, 32), stride=(16, 16), num_group=num_classes, no_bias=True,
            name='upsampling', attr={'lr_mult': '0.0'}, workspace=self.workspace)

        croped_score = mx.symbol.Crop(*[upsampling, data], offset=(8, 8), name='croped_score')

        softmax = mx.symbol.SoftmaxOutput(data=croped_score, normalization='valid', multi_output=True, use_ignore=True,
                                          ignore_label=255, name="softmax")

        return softmax

    def get_symbol(self, cfg, is_train=True):
        """
        return a generated symbol, it also need to be assigned to self.sym
        """

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES

        if is_train:
            self.sym = self.get_train_symbol(num_classes=num_classes)
        else:
            self.sym = self.get_test_symbol(num_classes=num_classes)

        return self.sym

    def init_weights(self, cfg, arg_params, aux_params):
        arg_params['fc6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc6_weight'])
        arg_params['fc6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc6_bias'])
        arg_params['score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['score_weight'])
        arg_params['score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['score_bias'])
        arg_params['upsampling_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['upsampling_weight'])

        init = mx.init.Initializer()
        init._init_bilinear('upsample_weight', arg_params['upsampling_weight'])

    def get_body(self, data, numberofparts, numberoflinks):
        sym_body = self.get_resnet_conv(data)
        r0 = mx.symbol.Convolution(data=sym_body, kernel=(3, 3),
                                   pad=(1, 1), num_filter=numberofparts, name="conv_last_heat",
                                   workspace=self.workspace)
        r1 = mx.symbol.Convolution(data=sym_body, kernel=(3, 3),
                                   pad=(1, 1), num_filter=numberoflinks * 2, name="conv_last_paf",
                                   workspace=self.workspace)


        conv4_3_CPM = mx.symbol.Convolution(name='conv4_3_CPM', data=sym_body, num_filter=256, pad=(1, 1),
                                            kernel=(3, 3),
                                            stride=(1, 1), no_bias=False)
        relu4_3_CPM = mx.symbol.Activation(name='relu4_3_CPM', data=conv4_3_CPM, act_type='relu')
        conv4_4_CPM = mx.symbol.Convolution(name='conv4_4_CPM', data=relu4_3_CPM, num_filter=128, pad=(1, 1),
                                            kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu4_4_CPM = mx.symbol.Activation(name='relu4_4_CPM', data=conv4_4_CPM, act_type='relu')
        conv5_1_CPM_L1 = mx.symbol.Convolution(name='conv5_1_CPM_L1', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu5_1_CPM_L1 = mx.symbol.Activation(name='relu5_1_CPM_L1', data=conv5_1_CPM_L1, act_type='relu')
        conv5_1_CPM_L2 = mx.symbol.Convolution(name='conv5_1_CPM_L2', data=relu4_4_CPM, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu5_1_CPM_L2 = mx.symbol.Activation(name='relu5_1_CPM_L2', data=conv5_1_CPM_L2, act_type='relu')
        conv5_2_CPM_L1 = mx.symbol.Convolution(name='conv5_2_CPM_L1', data=relu5_1_CPM_L1, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu5_2_CPM_L1 = mx.symbol.Activation(name='relu5_2_CPM_L1', data=conv5_2_CPM_L1, act_type='relu')
        conv5_2_CPM_L2 = mx.symbol.Convolution(name='conv5_2_CPM_L2', data=relu5_1_CPM_L2, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu5_2_CPM_L2 = mx.symbol.Activation(name='relu5_2_CPM_L2', data=conv5_2_CPM_L2, act_type='relu')
        conv5_3_CPM_L1 = mx.symbol.Convolution(name='conv5_3_CPM_L1', data=relu5_2_CPM_L1, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu5_3_CPM_L1 = mx.symbol.Activation(name='relu5_3_CPM_L1', data=conv5_3_CPM_L1, act_type='relu')
        conv5_3_CPM_L2 = mx.symbol.Convolution(name='conv5_3_CPM_L2', data=relu5_2_CPM_L2, num_filter=128, pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=False)
        relu5_3_CPM_L2 = mx.symbol.Activation(name='relu5_3_CPM_L2', data=conv5_3_CPM_L2, act_type='relu')
        conv5_4_CPM_L1 = mx.symbol.Convolution(name='conv5_4_CPM_L1', data=relu5_3_CPM_L1, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=False)
        relu5_4_CPM_L1 = mx.symbol.Activation(name='relu5_4_CPM_L1', data=conv5_4_CPM_L1, act_type='relu')
        conv5_4_CPM_L2 = mx.symbol.Convolution(name='conv5_4_CPM_L2', data=relu5_3_CPM_L2, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=False)
        relu5_4_CPM_L2 = mx.symbol.Activation(name='relu5_4_CPM_L2', data=conv5_4_CPM_L2, act_type='relu')
        conv5_5_CPM_L1 = mx.symbol.Convolution(name='conv5_5_CPM_L1', data=relu5_4_CPM_L1, num_filter=38, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=False)
        conv5_5_CPM_L2 = mx.symbol.Convolution(name='conv5_5_CPM_L2', data=relu5_4_CPM_L2, num_filter=19, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=False)
        concat_stage2 = mx.symbol.Concat(name='concat_stage2', *[conv5_5_CPM_L1, conv5_5_CPM_L2, relu4_4_CPM])
        Mconv1_stage2_L1 = mx.symbol.Convolution(name='Mconv1_stage2_L1', data=concat_stage2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage2_L1 = mx.symbol.Activation(name='Mrelu1_stage2_L1', data=Mconv1_stage2_L1, act_type='relu')
        Mconv1_stage2_L2 = mx.symbol.Convolution(name='Mconv1_stage2_L2', data=concat_stage2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage2_L2 = mx.symbol.Activation(name='Mrelu1_stage2_L2', data=Mconv1_stage2_L2, act_type='relu')
        Mconv2_stage2_L1 = mx.symbol.Convolution(name='Mconv2_stage2_L1', data=Mrelu1_stage2_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage2_L1 = mx.symbol.Activation(name='Mrelu2_stage2_L1', data=Mconv2_stage2_L1, act_type='relu')
        Mconv2_stage2_L2 = mx.symbol.Convolution(name='Mconv2_stage2_L2', data=Mrelu1_stage2_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage2_L2 = mx.symbol.Activation(name='Mrelu2_stage2_L2', data=Mconv2_stage2_L2, act_type='relu')
        Mconv3_stage2_L1 = mx.symbol.Convolution(name='Mconv3_stage2_L1', data=Mrelu2_stage2_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage2_L1 = mx.symbol.Activation(name='Mrelu3_stage2_L1', data=Mconv3_stage2_L1, act_type='relu')
        Mconv3_stage2_L2 = mx.symbol.Convolution(name='Mconv3_stage2_L2', data=Mrelu2_stage2_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage2_L2 = mx.symbol.Activation(name='Mrelu3_stage2_L2', data=Mconv3_stage2_L2, act_type='relu')
        Mconv4_stage2_L1 = mx.symbol.Convolution(name='Mconv4_stage2_L1', data=Mrelu3_stage2_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage2_L1 = mx.symbol.Activation(name='Mrelu4_stage2_L1', data=Mconv4_stage2_L1, act_type='relu')
        Mconv4_stage2_L2 = mx.symbol.Convolution(name='Mconv4_stage2_L2', data=Mrelu3_stage2_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage2_L2 = mx.symbol.Activation(name='Mrelu4_stage2_L2', data=Mconv4_stage2_L2, act_type='relu')
        Mconv5_stage2_L1 = mx.symbol.Convolution(name='Mconv5_stage2_L1', data=Mrelu4_stage2_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage2_L1 = mx.symbol.Activation(name='Mrelu5_stage2_L1', data=Mconv5_stage2_L1, act_type='relu')
        Mconv5_stage2_L2 = mx.symbol.Convolution(name='Mconv5_stage2_L2', data=Mrelu4_stage2_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage2_L2 = mx.symbol.Activation(name='Mrelu5_stage2_L2', data=Mconv5_stage2_L2, act_type='relu')
        Mconv6_stage2_L1 = mx.symbol.Convolution(name='Mconv6_stage2_L1', data=Mrelu5_stage2_L1, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage2_L1 = mx.symbol.Activation(name='Mrelu6_stage2_L1', data=Mconv6_stage2_L1, act_type='relu')
        Mconv6_stage2_L2 = mx.symbol.Convolution(name='Mconv6_stage2_L2', data=Mrelu5_stage2_L2, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage2_L2 = mx.symbol.Activation(name='Mrelu6_stage2_L2', data=Mconv6_stage2_L2, act_type='relu')
        Mconv7_stage2_L1 = mx.symbol.Convolution(name='Mconv7_stage2_L1', data=Mrelu6_stage2_L1, num_filter=38,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mconv7_stage2_L2 = mx.symbol.Convolution(name='Mconv7_stage2_L2', data=Mrelu6_stage2_L2, num_filter=19,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        concat_stage3 = mx.symbol.Concat(name='concat_stage3', *[Mconv7_stage2_L1, Mconv7_stage2_L2, relu4_4_CPM])
        Mconv1_stage3_L1 = mx.symbol.Convolution(name='Mconv1_stage3_L1', data=concat_stage3, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage3_L1 = mx.symbol.Activation(name='Mrelu1_stage3_L1', data=Mconv1_stage3_L1, act_type='relu')
        Mconv1_stage3_L2 = mx.symbol.Convolution(name='Mconv1_stage3_L2', data=concat_stage3, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage3_L2 = mx.symbol.Activation(name='Mrelu1_stage3_L2', data=Mconv1_stage3_L2, act_type='relu')
        Mconv2_stage3_L1 = mx.symbol.Convolution(name='Mconv2_stage3_L1', data=Mrelu1_stage3_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage3_L1 = mx.symbol.Activation(name='Mrelu2_stage3_L1', data=Mconv2_stage3_L1, act_type='relu')
        Mconv2_stage3_L2 = mx.symbol.Convolution(name='Mconv2_stage3_L2', data=Mrelu1_stage3_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage3_L2 = mx.symbol.Activation(name='Mrelu2_stage3_L2', data=Mconv2_stage3_L2, act_type='relu')
        Mconv3_stage3_L1 = mx.symbol.Convolution(name='Mconv3_stage3_L1', data=Mrelu2_stage3_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage3_L1 = mx.symbol.Activation(name='Mrelu3_stage3_L1', data=Mconv3_stage3_L1, act_type='relu')
        Mconv3_stage3_L2 = mx.symbol.Convolution(name='Mconv3_stage3_L2', data=Mrelu2_stage3_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage3_L2 = mx.symbol.Activation(name='Mrelu3_stage3_L2', data=Mconv3_stage3_L2, act_type='relu')
        Mconv4_stage3_L1 = mx.symbol.Convolution(name='Mconv4_stage3_L1', data=Mrelu3_stage3_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage3_L1 = mx.symbol.Activation(name='Mrelu4_stage3_L1', data=Mconv4_stage3_L1, act_type='relu')
        Mconv4_stage3_L2 = mx.symbol.Convolution(name='Mconv4_stage3_L2', data=Mrelu3_stage3_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage3_L2 = mx.symbol.Activation(name='Mrelu4_stage3_L2', data=Mconv4_stage3_L2, act_type='relu')
        Mconv5_stage3_L1 = mx.symbol.Convolution(name='Mconv5_stage3_L1', data=Mrelu4_stage3_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage3_L1 = mx.symbol.Activation(name='Mrelu5_stage3_L1', data=Mconv5_stage3_L1, act_type='relu')
        Mconv5_stage3_L2 = mx.symbol.Convolution(name='Mconv5_stage3_L2', data=Mrelu4_stage3_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage3_L2 = mx.symbol.Activation(name='Mrelu5_stage3_L2', data=Mconv5_stage3_L2, act_type='relu')
        Mconv6_stage3_L1 = mx.symbol.Convolution(name='Mconv6_stage3_L1', data=Mrelu5_stage3_L1, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage3_L1 = mx.symbol.Activation(name='Mrelu6_stage3_L1', data=Mconv6_stage3_L1, act_type='relu')
        Mconv6_stage3_L2 = mx.symbol.Convolution(name='Mconv6_stage3_L2', data=Mrelu5_stage3_L2, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage3_L2 = mx.symbol.Activation(name='Mrelu6_stage3_L2', data=Mconv6_stage3_L2, act_type='relu')
        Mconv7_stage3_L1 = mx.symbol.Convolution(name='Mconv7_stage3_L1', data=Mrelu6_stage3_L1, num_filter=38,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mconv7_stage3_L2 = mx.symbol.Convolution(name='Mconv7_stage3_L2', data=Mrelu6_stage3_L2, num_filter=19,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        concat_stage4 = mx.symbol.Concat(name='concat_stage4', *[Mconv7_stage3_L1, Mconv7_stage3_L2, relu4_4_CPM])
        Mconv1_stage4_L1 = mx.symbol.Convolution(name='Mconv1_stage4_L1', data=concat_stage4, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage4_L1 = mx.symbol.Activation(name='Mrelu1_stage4_L1', data=Mconv1_stage4_L1, act_type='relu')
        Mconv1_stage4_L2 = mx.symbol.Convolution(name='Mconv1_stage4_L2', data=concat_stage4, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage4_L2 = mx.symbol.Activation(name='Mrelu1_stage4_L2', data=Mconv1_stage4_L2, act_type='relu')
        Mconv2_stage4_L1 = mx.symbol.Convolution(name='Mconv2_stage4_L1', data=Mrelu1_stage4_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage4_L1 = mx.symbol.Activation(name='Mrelu2_stage4_L1', data=Mconv2_stage4_L1, act_type='relu')
        Mconv2_stage4_L2 = mx.symbol.Convolution(name='Mconv2_stage4_L2', data=Mrelu1_stage4_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage4_L2 = mx.symbol.Activation(name='Mrelu2_stage4_L2', data=Mconv2_stage4_L2, act_type='relu')
        Mconv3_stage4_L1 = mx.symbol.Convolution(name='Mconv3_stage4_L1', data=Mrelu2_stage4_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage4_L1 = mx.symbol.Activation(name='Mrelu3_stage4_L1', data=Mconv3_stage4_L1, act_type='relu')
        Mconv3_stage4_L2 = mx.symbol.Convolution(name='Mconv3_stage4_L2', data=Mrelu2_stage4_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage4_L2 = mx.symbol.Activation(name='Mrelu3_stage4_L2', data=Mconv3_stage4_L2, act_type='relu')
        Mconv4_stage4_L1 = mx.symbol.Convolution(name='Mconv4_stage4_L1', data=Mrelu3_stage4_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage4_L1 = mx.symbol.Activation(name='Mrelu4_stage4_L1', data=Mconv4_stage4_L1, act_type='relu')
        Mconv4_stage4_L2 = mx.symbol.Convolution(name='Mconv4_stage4_L2', data=Mrelu3_stage4_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage4_L2 = mx.symbol.Activation(name='Mrelu4_stage4_L2', data=Mconv4_stage4_L2, act_type='relu')
        Mconv5_stage4_L1 = mx.symbol.Convolution(name='Mconv5_stage4_L1', data=Mrelu4_stage4_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage4_L1 = mx.symbol.Activation(name='Mrelu5_stage4_L1', data=Mconv5_stage4_L1, act_type='relu')
        Mconv5_stage4_L2 = mx.symbol.Convolution(name='Mconv5_stage4_L2', data=Mrelu4_stage4_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage4_L2 = mx.symbol.Activation(name='Mrelu5_stage4_L2', data=Mconv5_stage4_L2, act_type='relu')
        Mconv6_stage4_L1 = mx.symbol.Convolution(name='Mconv6_stage4_L1', data=Mrelu5_stage4_L1, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage4_L1 = mx.symbol.Activation(name='Mrelu6_stage4_L1', data=Mconv6_stage4_L1, act_type='relu')
        Mconv6_stage4_L2 = mx.symbol.Convolution(name='Mconv6_stage4_L2', data=Mrelu5_stage4_L2, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage4_L2 = mx.symbol.Activation(name='Mrelu6_stage4_L2', data=Mconv6_stage4_L2, act_type='relu')
        Mconv7_stage4_L1 = mx.symbol.Convolution(name='Mconv7_stage4_L1', data=Mrelu6_stage4_L1, num_filter=38,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mconv7_stage4_L2 = mx.symbol.Convolution(name='Mconv7_stage4_L2', data=Mrelu6_stage4_L2, num_filter=19,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        concat_stage5 = mx.symbol.Concat(name='concat_stage5', *[Mconv7_stage4_L1, Mconv7_stage4_L2, relu4_4_CPM])
        Mconv1_stage5_L1 = mx.symbol.Convolution(name='Mconv1_stage5_L1', data=concat_stage5, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage5_L1 = mx.symbol.Activation(name='Mrelu1_stage5_L1', data=Mconv1_stage5_L1, act_type='relu')
        Mconv1_stage5_L2 = mx.symbol.Convolution(name='Mconv1_stage5_L2', data=concat_stage5, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage5_L2 = mx.symbol.Activation(name='Mrelu1_stage5_L2', data=Mconv1_stage5_L2, act_type='relu')
        Mconv2_stage5_L1 = mx.symbol.Convolution(name='Mconv2_stage5_L1', data=Mrelu1_stage5_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage5_L1 = mx.symbol.Activation(name='Mrelu2_stage5_L1', data=Mconv2_stage5_L1, act_type='relu')
        Mconv2_stage5_L2 = mx.symbol.Convolution(name='Mconv2_stage5_L2', data=Mrelu1_stage5_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage5_L2 = mx.symbol.Activation(name='Mrelu2_stage5_L2', data=Mconv2_stage5_L2, act_type='relu')
        Mconv3_stage5_L1 = mx.symbol.Convolution(name='Mconv3_stage5_L1', data=Mrelu2_stage5_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage5_L1 = mx.symbol.Activation(name='Mrelu3_stage5_L1', data=Mconv3_stage5_L1, act_type='relu')
        Mconv3_stage5_L2 = mx.symbol.Convolution(name='Mconv3_stage5_L2', data=Mrelu2_stage5_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage5_L2 = mx.symbol.Activation(name='Mrelu3_stage5_L2', data=Mconv3_stage5_L2, act_type='relu')
        Mconv4_stage5_L1 = mx.symbol.Convolution(name='Mconv4_stage5_L1', data=Mrelu3_stage5_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage5_L1 = mx.symbol.Activation(name='Mrelu4_stage5_L1', data=Mconv4_stage5_L1, act_type='relu')
        Mconv4_stage5_L2 = mx.symbol.Convolution(name='Mconv4_stage5_L2', data=Mrelu3_stage5_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage5_L2 = mx.symbol.Activation(name='Mrelu4_stage5_L2', data=Mconv4_stage5_L2, act_type='relu')
        Mconv5_stage5_L1 = mx.symbol.Convolution(name='Mconv5_stage5_L1', data=Mrelu4_stage5_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage5_L1 = mx.symbol.Activation(name='Mrelu5_stage5_L1', data=Mconv5_stage5_L1, act_type='relu')
        Mconv5_stage5_L2 = mx.symbol.Convolution(name='Mconv5_stage5_L2', data=Mrelu4_stage5_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage5_L2 = mx.symbol.Activation(name='Mrelu5_stage5_L2', data=Mconv5_stage5_L2, act_type='relu')
        Mconv6_stage5_L1 = mx.symbol.Convolution(name='Mconv6_stage5_L1', data=Mrelu5_stage5_L1, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage5_L1 = mx.symbol.Activation(name='Mrelu6_stage5_L1', data=Mconv6_stage5_L1, act_type='relu')
        Mconv6_stage5_L2 = mx.symbol.Convolution(name='Mconv6_stage5_L2', data=Mrelu5_stage5_L2, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage5_L2 = mx.symbol.Activation(name='Mrelu6_stage5_L2', data=Mconv6_stage5_L2, act_type='relu')
        Mconv7_stage5_L1 = mx.symbol.Convolution(name='Mconv7_stage5_L1', data=Mrelu6_stage5_L1, num_filter=38,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mconv7_stage5_L2 = mx.symbol.Convolution(name='Mconv7_stage5_L2', data=Mrelu6_stage5_L2, num_filter=19,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        concat_stage6 = mx.symbol.Concat(name='concat_stage6', *[Mconv7_stage5_L1, Mconv7_stage5_L2, relu4_4_CPM])
        Mconv1_stage6_L1 = mx.symbol.Convolution(name='Mconv1_stage6_L1', data=concat_stage6, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage6_L1 = mx.symbol.Activation(name='Mrelu1_stage6_L1', data=Mconv1_stage6_L1, act_type='relu')
        Mconv1_stage6_L2 = mx.symbol.Convolution(name='Mconv1_stage6_L2', data=concat_stage6, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu1_stage6_L2 = mx.symbol.Activation(name='Mrelu1_stage6_L2', data=Mconv1_stage6_L2, act_type='relu')
        Mconv2_stage6_L1 = mx.symbol.Convolution(name='Mconv2_stage6_L1', data=Mrelu1_stage6_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage6_L1 = mx.symbol.Activation(name='Mrelu2_stage6_L1', data=Mconv2_stage6_L1, act_type='relu')
        Mconv2_stage6_L2 = mx.symbol.Convolution(name='Mconv2_stage6_L2', data=Mrelu1_stage6_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu2_stage6_L2 = mx.symbol.Activation(name='Mrelu2_stage6_L2', data=Mconv2_stage6_L2, act_type='relu')
        Mconv3_stage6_L1 = mx.symbol.Convolution(name='Mconv3_stage6_L1', data=Mrelu2_stage6_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage6_L1 = mx.symbol.Activation(name='Mrelu3_stage6_L1', data=Mconv3_stage6_L1, act_type='relu')
        Mconv3_stage6_L2 = mx.symbol.Convolution(name='Mconv3_stage6_L2', data=Mrelu2_stage6_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu3_stage6_L2 = mx.symbol.Activation(name='Mrelu3_stage6_L2', data=Mconv3_stage6_L2, act_type='relu')
        Mconv4_stage6_L1 = mx.symbol.Convolution(name='Mconv4_stage6_L1', data=Mrelu3_stage6_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage6_L1 = mx.symbol.Activation(name='Mrelu4_stage6_L1', data=Mconv4_stage6_L1, act_type='relu')
        Mconv4_stage6_L2 = mx.symbol.Convolution(name='Mconv4_stage6_L2', data=Mrelu3_stage6_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu4_stage6_L2 = mx.symbol.Activation(name='Mrelu4_stage6_L2', data=Mconv4_stage6_L2, act_type='relu')
        Mconv5_stage6_L1 = mx.symbol.Convolution(name='Mconv5_stage6_L1', data=Mrelu4_stage6_L1, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage6_L1 = mx.symbol.Activation(name='Mrelu5_stage6_L1', data=Mconv5_stage6_L1, act_type='relu')
        Mconv5_stage6_L2 = mx.symbol.Convolution(name='Mconv5_stage6_L2', data=Mrelu4_stage6_L2, num_filter=128,
                                                 pad=(3, 3), kernel=(7, 7), stride=(1, 1), no_bias=False)
        Mrelu5_stage6_L2 = mx.symbol.Activation(name='Mrelu5_stage6_L2', data=Mconv5_stage6_L2, act_type='relu')
        Mconv6_stage6_L1 = mx.symbol.Convolution(name='Mconv6_stage6_L1', data=Mrelu5_stage6_L1, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage6_L1 = mx.symbol.Activation(name='Mrelu6_stage6_L1', data=Mconv6_stage6_L1, act_type='relu')
        Mconv6_stage6_L2 = mx.symbol.Convolution(name='Mconv6_stage6_L2', data=Mrelu5_stage6_L2, num_filter=128,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mrelu6_stage6_L2 = mx.symbol.Activation(name='Mrelu6_stage6_L2', data=Mconv6_stage6_L2, act_type='relu')
        Mconv7_stage6_L1 = mx.symbol.Convolution(name='Mconv7_stage6_L1', data=Mrelu6_stage6_L1, num_filter=38,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
        Mconv7_stage6_L2 = mx.symbol.Convolution(name='Mconv7_stage6_L2', data=Mrelu6_stage6_L2, num_filter=19,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)

        # return conv5_5_CPM_L2

        heatmaps_predicts = [ conv5_5_CPM_L2, Mconv7_stage2_L2, Mconv7_stage3_L2, Mconv7_stage4_L2,
                             Mconv7_stage5_L2, Mconv7_stage6_L2]
        pafmaps_predicts = [conv5_5_CPM_L1, Mconv7_stage2_L1, Mconv7_stage3_L1, Mconv7_stage4_L1,
                            Mconv7_stage5_L1, Mconv7_stage6_L1]
        heatmaps_predicts = [r0]
        pafmaps_predicts = [r1]
        return heatmaps_predicts, pafmaps_predicts


def get_symbol(is_train=True, numberofparts=18, numberoflinks=19):
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")
    # heatmap, partaffinity, heatweight, vecweight = slice_label(label, numberofparts, numberoflinks)
    heatmap = mx.symbol.Variable(name = "heatmaplabel")
#     partaffinity = mx.symbol.Variable(name = "partaffinityglabel")
#     loss_mask = mx.symbol.Variable(name = "loss_mask")
    Sym = resnet_v1_101_deeplab()
    heat_ps, paf_ps = Sym.get_body(data, numberofparts, numberoflinks)
    losses = []
    if is_train:
        for heat_p, paf_p in zip(heat_ps, paf_ps):
            heat_p = mx.sym.sigmoid(heat_p)
#             loss1 = (heatmap-heat_p)**2
            loss1 = heatmap * mx.sym.log(heat_p+1e-7) + (1-heatmap)*mx.sym.log(1-heat_p+1e-7)
            loss1 = loss1 * -1
            losses.append(loss1)
        # return mx.symbol.Group([mx.symbol.MakeLoss(loss1),mx.symbol.MakeLoss(loss2),mx.symbol.BlockGrad(heat_p),mx.symbol.BlockGrad(paf_p)])
        return mx.sym.Group([mx.sym.MakeLoss(x) for x in losses] + [mx.sym.BlockGrad(heat_p)])
    else:
        return mx.symbol.Group([mx.sym.sigmoid(x) for x in heat_ps] + paf_ps)
        # return mx.symbol.Group(heat_ps + paf_ps)


if __name__ == "__main__":
    import pprint
    pprint.pprint(get_symbol(is_train=False).get_internals().list_outputs())
    mx.visualization.plot_network(get_symbol(is_train=True),shape = {"data":(1,3,96*2,96)}).view()
