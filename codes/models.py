#coding:utf-8
import mxnet as mx
import numpy as np
import sys

sys.dont_write_bytecode = True

from config import *

def deconv_net(is_train):
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')
    lossfactor = mx.symbol.Variable('lossfactor', shape=(1), init=mx.init.Uniform(scale=.1))

    conv_1_weight = mx.symbol.Variable('arg:conv_1_weight')
    conv_2_weight = mx.symbol.Variable('arg:conv_2_weight')
    conv_3_weight = mx.symbol.Variable('arg:conv_3_weight')
    conv_4_weight = mx.symbol.Variable('arg:conv_4_weight')
    conv_5_weight = mx.symbol.Variable('arg:conv_5_weight')
    conv_6_weight = mx.symbol.Variable('arg:conv_6_weight')
    conv_7_weight = mx.symbol.Variable('arg:conv_7_weight')
    conv_8_weight = mx.symbol.Variable('arg:conv_8_weight')
    conv_9_weight = mx.symbol.Variable('arg:conv_9_weight')
    conv_10_weight = mx.symbol.Variable('arg:conv_10_weight')
    conv_11_weight = mx.symbol.Variable('arg:conv_11_weight')
    conv_12_weight = mx.symbol.Variable('arg:conv_12_weight')
    conv_13_weight = mx.symbol.Variable('arg:conv_13_weight')

    # decv_11_weight = mx.symbol.Variable('arg:decv_11_weight')

    conv_1_bias = mx.symbol.Variable('arg:conv_1_bias')
    conv_2_bias = mx.symbol.Variable('arg:conv_2_bias')
    conv_3_bias = mx.symbol.Variable('arg:conv_3_bias')
    conv_4_bias = mx.symbol.Variable('arg:conv_4_bias')
    conv_5_bias = mx.symbol.Variable('arg:conv_5_bias')
    conv_6_bias = mx.symbol.Variable('arg:conv_6_bias')
    conv_7_bias = mx.symbol.Variable('arg:conv_7_bias')
    conv_8_bias = mx.symbol.Variable('arg:conv_8_bias')
    conv_9_bias = mx.symbol.Variable('arg:conv_9_bias')
    conv_10_bias = mx.symbol.Variable('arg:conv_10_bias')
    conv_11_bias = mx.symbol.Variable('arg:conv_11_bias')
    conv_12_bias = mx.symbol.Variable('arg:conv_12_bias')
    conv_13_bias = mx.symbol.Variable('arg:conv_13_bias')

    # 3-64 640*480
    conv_1 = mx.symbol.Convolution(data=data, name='conv1', num_filter=64, kernel=(3,3), pad=(1,1), weight=conv_1_weight, bias=conv_1_bias)
    norm_1 = mx.symbol.BatchNorm(data=conv_1, name='norm1')
    actv_1 = mx.symbol.Activation(data=norm_1, name='actv1', act_type='relu')
    conv_2 = mx.symbol.Convolution(data=actv_1, name='conv2', num_filter=64, kernel=(3,3), pad=(1,1), weight=conv_2_weight, bias=conv_2_bias)
    norm_2 = mx.symbol.BatchNorm(data=conv_2, name='norm2')
    actv_2 = mx.symbol.Activation(data=norm_2, name='actv2', act_type='relu')
    pool_2 = mx.symbol.Pooling(data=actv_2, name='pool2', stride=(2,2), kernel=(2,2), pool_type='max')

    # 64-128 320*240
    conv_3 = mx.symbol.Convolution(data=pool_2, name='conv3', num_filter=128, kernel=(3,3), pad=(1,1), weight=conv_3_weight, bias=conv_3_bias)
    norm_3 = mx.symbol.BatchNorm(data=conv_3, name='norm3')
    actv_3 = mx.symbol.Activation(data=norm_3, name='actv3', act_type='relu')
    conv_4 = mx.symbol.Convolution(data=actv_3, name='conv4', num_filter=128, kernel=(3,3), pad=(1,1), weight=conv_4_weight, bias=conv_4_bias)
    norm_4 = mx.symbol.BatchNorm(data=conv_4, name='norm4')
    actv_4 = mx.symbol.Activation(data=norm_4, name='actv4', act_type='relu')
    pool_4 = mx.symbol.Pooling(data=actv_4, name='pool4', stride=(2,2), kernel=(2,2), pool_type='max')

    # 128-256 160*120
    conv_5 = mx.symbol.Convolution(data=pool_4, name='conv5', num_filter=256, kernel=(3,3), pad=(1,1), weight=conv_5_weight, bias=conv_5_bias)
    norm_5 = mx.symbol.BatchNorm(data=conv_5, name='norm5')
    actv_5 = mx.symbol.Activation(data=norm_5, name='act5', act_type='relu')
    conv_6 = mx.symbol.Convolution(data=actv_5, name='conv6', num_filter=256, kernel=(3,3), pad=(1,1), weight=conv_6_weight, bias=conv_6_bias)
    norm_6 = mx.symbol.BatchNorm(data=conv_6, name='norm6')
    actv_6 = mx.symbol.Activation(data=norm_6, name='act6', act_type='relu')
    conv_7 = mx.symbol.Convolution(data=actv_6, name='conv7', num_filter=256, kernel=(3,3), pad=(1,1), weight=conv_7_weight, bias=conv_7_bias)
    norm_7 = mx.symbol.BatchNorm(data=conv_7, name='norm7')
    actv_7 = mx.symbol.Activation(data=norm_7, name='act7', act_type='relu')
    pool_7 = mx.symbol.Pooling(data=actv_7, name='pool7', stride=(2,2), kernel=(2,2), pool_type='max')

    # 256-512 80*60
    conv_8 = mx.symbol.Convolution(data=pool_7, name='conv8', num_filter=512, kernel=(3,3), pad=(2,2), dilate=(2,2), weight=conv_8_weight, bias=conv_8_bias)
    norm_8 = mx.symbol.BatchNorm(data=conv_8, name='norm8')
    actv_8 = mx.symbol.Activation(data=norm_8, name='act8', act_type='relu')
    conv_9 = mx.symbol.Convolution(data=actv_8, name='conv9', num_filter=512, kernel=(3,3), pad=(2,2), dilate=(2,2), weight=conv_9_weight, bias=conv_9_bias)
    norm_9 = mx.symbol.BatchNorm(data=conv_9, name='norm9')
    actv_9 = mx.symbol.Activation(data=norm_9, name='act9', act_type='relu')
    conv_10 = mx.symbol.Convolution(data=actv_9, name='conv10', num_filter=512, kernel=(3,3), pad=(2,2), dilate=(2,2), weight=conv_10_weight, bias=conv_10_bias)
    norm_10 = mx.symbol.BatchNorm(data=conv_10, name='norm10')
    actv_10 = mx.symbol.Activation(data=norm_10, name='act10', act_type='relu')
    # pool_10 = mx.symbol.Pooling(data=actv_10, name='pool10', stride=(2,2), kernel=(2,2), pool_type='max')

    # 512-512 40*30
    conv_11 = mx.symbol.Convolution(data=actv_10, name='conv11', num_filter=512, kernel=(3,3), pad=(2,2), dilate=(2,2), weight=conv_11_weight, bias=conv_11_bias)
    norm_11 = mx.symbol.BatchNorm(data=conv_11, name='norm11')
    actv_11 = mx.symbol.Activation(data=norm_11, name='act11', act_type='relu')
    conv_12 = mx.symbol.Convolution(data=actv_11, name='conv12', num_filter=512, kernel=(3,3), pad=(2,2), dilate=(2,2), weight=conv_12_weight, bias=conv_12_bias)
    norm_12 = mx.symbol.BatchNorm(data=conv_12, name='norm12')
    actv_12 = mx.symbol.Activation(data=norm_12, name='act12', act_type='relu')
    conv_13 = mx.symbol.Convolution(data=actv_12, name='conv13', num_filter=512, kernel=(3,3), pad=(2,2), dilate=(2,2), weight=conv_13_weight, bias=conv_13_bias)
    norm_13 = mx.symbol.BatchNorm(data=conv_13, name='norm13')
    actv_13 = mx.symbol.Activation(data=norm_13, name='act13', act_type='relu')
    # pool_13 = mx.symbol.Pooling(data=actv_13, name='pool13', stride=(1,1), kernel=(3,3), pad=(1,1), pool_type='avg')

    block = mx.symbol.BlockGrad(data=actv_13, name='block')

    conv_h = mx.symbol.Convolution(data=actv_13, name='convh', num_filter=32, kernel=(3,3), dilate=(4,4), pad=(4,4))
    norm_h = mx.symbol.BatchNorm(data=conv_h, name='normh')
    actv_h = mx.symbol.Activation(data=norm_h, name='acth', act_type='relu')

    concat = mx.symbol.concat(block, actv_h)

    conv_i = mx.symbol.Convolution(data=concat, name='convi', num_filter=128, kernel=(1,1))
    norm_i = mx.symbol.BatchNorm(data=conv_i, name='normi')
    actv_i = mx.symbol.Activation(data=norm_i, name='acti', act_type='relu')

    # 512-512 40*30
    # decv_d = mx.symbol.Deconvolution(data=actv_i, name='decvd', num_filter=16, kernel=(4,4), stride=(4,4))
    # norm_d = mx.symbol.BatchNorm(data=decv_d, name='normd')
    # actv_d = mx.symbol.Activation(data=norm_d, name='actd', act_type='relu')
    
    conv_o = mx.symbol.Convolution(data=actv_i, name='convo', num_filter=1, kernel=(1,1))
    norm_o = mx.symbol.BatchNorm(data=conv_o, name='normo')
    actv_o = mx.symbol.Activation(data=norm_o, name='acto', act_type='relu')

    max_ = mx.symbol.max(data=actv_o, axis=())
    min_ = mx.symbol.min(data=actv_o, axis=())
    tff_ = mx.symbol.full(shape=(1, OU_W, OU_H), val=250.0)
    up__ = mx.symbol.broadcast_sub(lhs=actv_o, rhs=min_)
    down = mx.symbol.broadcast_sub(lhs=max_, rhs=min_)
    div_ = mx.symbol.broadcast_div(lhs=up__, rhs=down)
    out = mx.symbol.broadcast_mul(lhs=tff_, rhs=div_)

    opt1 = mx.symbol.broadcast_sub(out,label)
    # opt2 = mx.symbol.broadcast_div(mx.symbol.abs(mx.symbol.broadcast_sub(out,label)),mx.symbol.full(shape=(1, OU_W, OU_H), val=2.0))
    opt2 = mx.symbol.broadcast_div(mx.symbol.abs(mx.symbol.broadcast_sub(out,label)),lossfactor)
    opt3 = mx.symbol.square(mx.symbol.broadcast_add(opt1,opt2))
    loss = mx.symbol.MakeLoss(opt3)
    # loss = mx.symbol.LinearRegressionOutput(data=out, label=label, name='loss')
    if is_train:
        return loss
    else:
        return out


