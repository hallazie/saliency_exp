#coding:utf-8
import mxnet as mx
import numpy as np
import traceback
import logging
import dataiter
import sys

from collections import namedtuple
from PIL import Image
from config import *

BATCH_SIZE = 1
SAMPLE_CNT = 10000
VALID_CNT = 5000
EPOCH_CNT = 100
IN_W = 640
IN_H = 480
OU_W = 40
OU_H = 30
DATA_PATH = '../../datasets/salicon/Trn'
LABEL_PATH = '../../datasets/salicon/Trn_Map'

Batch = namedtuple('Batch', ['data'])

def test():
	diter = dataiter.SaliencyIter()
	symbol = deconv_net()
	arg_names = symbol.list_arguments()
	arg_shapes, output_shapes, aux_shapes = symbol.infer_shape(data=(1,3,IN_W,IN_H))
	aux_params = {}
	for name,shape in zip(arg_names,arg_shapes):
		print name+' : '+str(shape)

	model = mx.mod.Module(symbol=symbol, context=mx.cpu(), data_names=('data',), label_names=('label',))
	model.bind(data_shapes=diter.provide_data, label_shapes=model._label_shapes)

	model.init_params()
	arg_params, aux_params = model.get_params()
	arg_params_load = mx.nd.load('../params/vgg/vgg16-pre.params')
	for k in arg_params_load:
		arg_params[k] = arg_params_load[k]
	model.set_params(arg_params, aux_params, allow_missing=True)

	fname_list = []
	for _,_,f in os.walk(DATA_PATH):
		fname_list.extend(f)
	random.shuffle(fname_list)
	for i in range(1):
		try:
			img = np.array(Image.open(DATA_PATH+'/'+fname_list[i]).resize((IN_W,IN_H)))
			img = np.swapaxes(img,0,2)
			model.forward(Batch([mx.nd.array(img).reshape((1,3,640,480))]))
			pred = model.get_outputs()[0].asnumpy()
			# print pred.shape
			# print img.shape
			for j in range(512):
				Image.fromarray(pred[0][0][j].transpose().astype('uint8')).resize((640,480)).save('../pred/'+fname_list[i].split('.')[0]+str(j)+'_pred.jpg')
			Image.fromarray(np.swapaxes(img,0,2).astype('uint8')).resize((640,480)).save('../pred/'+fname_list[i])
			print '%sth img prediction finished'%i
		except:
			traceback.print_exc()
			return 0

def deconv_net():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')

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
    pool_10 = mx.symbol.Pooling(data=actv_10, name='pool10', stride=(2,2), kernel=(2,2), pool_type='max')

    # # 512-512 40*30
    # decv_11 = mx.symbol.Deconvolution(data=pool_10, name='decv11', num_filter=512, kernel=(8,8), stride=(8,8))
    # norm_11 = mx.symbol.BatchNorm(data=decv_11, name='norm11')
    # actv_11 = mx.symbol.Activation(data=norm_11, name='act11', act_type='relu')
    
    # conv_o = mx.symbol.Convolution(data=actv_11, name='convo', num_filter=1, kernel=(1,1))
    # norm_o = mx.symbol.BatchNorm(data=conv_o, name='normo')
    # actv_o = mx.symbol.Activation(data=norm_o, name='acto', act_type='relu')

    out = mx.symbol.MAERegressionOutput(data=pool_10, label=label, name='out')
    return out

if __name__ == '__main__':
	test()
