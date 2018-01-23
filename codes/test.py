#coding:utf-8
import mxnet as mx
import numpy as np
import models
import traceback
import logging
import dataiter
import sys

from collections import namedtuple
from PIL import Image, ImageFilter
from config import *

Batch = namedtuple('Batch', ['data'])
img_path = DATA_PATH

def test():
	epoch = int(sys.argv[1])
	diter = dataiter.SaliencyIter()
	symbol = models.deconv_net(False)

	arg_names = symbol.list_arguments()
	aux_names = symbol.list_auxiliary_states()
	# arg_shapes, output_shapes, aux_shapes = symbol.infer_shape(data=(1,3,IN_W,IN_H))
	# aux_params = {}
	# for name,shape in zip(arg_names,arg_shapes):
	# 	print name+' : '+str(shape)

	model = mx.mod.Module(symbol=symbol, context=mx.cpu(), label_names=None)
	model.bind(for_training=False, data_shapes=diter._provide_data)

	sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL_PREFIX, epoch)
	try:
		del arg_params['lossfactor']
	except:
		pass
	model.set_params(arg_params, aux_params, allow_missing=True)

	# arg_params, aux_params = {}, {}
	# arg_params_load = mx.nd.load(PARAMS_PATH)
	# for k in arg_params_load:
	#	if k in arg_names:
	#		arg_params[k] = arg_params_load[k]
	#	elif k in aux_names:
	#		aux_params[k] = arg_params_load[k]
	#	else:
	##  		print 'invalid param %s with value %s...'%(k, arg_params_load[k][0:5])
	#		pass
	# model.set_params(arg_params, aux_params, allow_missing=True)

	fname_list = []
	for _,_,f in os.walk(img_path):
		fname_list.extend(f)
	random.shuffle(fname_list)
	for i in range(int(sys.argv[2])):
		try:
			img = Image.open(img_path+'/'+fname_list[i])
			w, h = img.size
			img = np.array(img)
			lbl = np.array(Image.open(LABEL_PATH+'/'+fname_list[i].split('.')[0]+POST_FIX).resize((w,h)))
			img = np.swapaxes(img,0,2)
			model.forward(Batch([mx.nd.array(img).reshape((1,3,w,h))]),is_train=False)
			pred = model.get_outputs()[0].asnumpy()[0][0]
			pred = 254*(pred-np.amin(pred))/(np.amax(pred)-np.amin(pred))
			Image.fromarray(pred.transpose().astype('uint8')).resize((w,h)).convert('L').filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.GaussianBlur(5)).save('../pred/'+fname_list[i].split('.')[0]+'.jpg')
			# Image.fromarray(pred.transpose().astype('uint8')).resize((w,h)).save('../pred/'+fname_list[i].split('.')[0]+'_pred.jpg')
			# Image.fromarray(np.swapaxes(img,0,2).astype('uint8')).resize((w,h)).save('../pred/'+fname_list[i])
			# Image.fromarray(lbl.astype('uint8')).resize((w,h)).save('../pred/'+fname_list[i].split('.')[0]+'_label.jpg')
			print '%sth img prediction finished with pixel range of %s~%s'%(i,np.amin(pred),np.amax(pred))
		except:
			traceback.print_exc()
			return 0
if __name__ == '__main__':
	test()
