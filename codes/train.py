#coding:utf-8
import mxnet as mx
import numpy as np
import models
import dataiter
import traceback
import logging
import models

from PIL import Image

from config import *

logging.getLogger().setLevel(logging.DEBUG)

def train():
	
	diter = dataiter.SaliencyIter()
	# viter = dataiter.SaliencyValidateIter()
	symbol = models.deconv_net(True)
	arg_names = symbol.list_arguments()

	model = mx.mod.Module(symbol=symbol, context=mx.gpu(), data_names=('data',), label_names=('label',))
	model.bind(data_shapes=diter.provide_data, label_shapes=diter.provide_label)
	
	# model.init_params(initializer=mx.init.Uniform(scale=.1))
	# arg_params, aux_params = model.get_params()
	# arg_params_load = mx.nd.load(VGG_PATH)
	# for k in arg_params_load:
	# 	if k in arg_names:
	# 		arg_params[k] = arg_params_load[k]
	# model.set_params(arg_params, aux_params, allow_missing=True)

	sym, arg_params, aux_params = mx.model.load_checkpoint(MODEL_PREFIX, 1)
	model.set_params(arg_params, aux_params, allow_missing=True)

	model.fit(
		diter,
		optimizer = 'adam',
		optimizer_params = {'learning_rate':0.005},
		eval_metric = 'mse',
		batch_end_callback = mx.callback.Speedometer(BATCH_SIZE, 10),
		epoch_end_callback = mx.callback.do_checkpoint(MODEL_PREFIX, 1),
		num_epoch = 100,
		)
	
	# fname_list = []
	# for _,_,f in os.walk(DATA_PATH):
	# 	fname_list.extend(f)
	# random.shuffle(fname_list)
	# for e in range(EPOCH_CNT):
	# 	diter.reset()
	# 	metric.reset()
	# 	idx = 0
	# 	for batch in diter:
	# 		idx += 1
	# 		model.forward(mx.io.DataBatch(data=batch.data, label=batch.label))
	# 		if e == 0:
	# 			Image.fromarray(np.swapaxes(batch.data[0][0].asnumpy(),0,2).astype('uint8')).resize((640,480)).save('../pred/'+fname_list[0])
	# 			Image.fromarray(np.swapaxes(batch.label[0][0][0].asnumpy(),0,1).astype('uint8')).save('../pred/'+fname_list[0].split('.')[0]+'_label.jpg')
	# 		model.update_metric(metric, batch.label)
	# 		model.backward()
	# 		model.update()
	# 		print('Epoch %d, Batch %d, Training MSE Loss: %s' % (e, idx, metric.get()))

if __name__ == '__main__':
	train()
