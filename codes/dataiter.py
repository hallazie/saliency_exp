import mxnet as mx
import numpy as np
import os
import sys

sys.dont_write_bytecode = True

from config import *
from PIL import Image

class SaliencyIter(mx.io.DataIter):
    def __init__(self):
        self.data_path = DATA_PATH
        self.label_path = LABEL_PATH
        self.batch_size = BATCH_SIZE
        self.num_sample = SAMPLE_CNT
        self.data_width = IN_W
        self.data_height = IN_H
        self.label_width = OU_W
        self.label_height = OU_H
        self._provide_data = zip(['data'], [(self.batch_size,3,self.data_width,self.data_height)])
        self._provide_label = zip(['label'], [(self.batch_size,1,self.label_width,self.label_height)])
        self.num_batches = self.num_sample/self.batch_size

        self.fname_list = self.get_lst()
        self.data_gen = self.data_iter(self.fname_list)
        self.label_gen = self.label_iter(self.fname_list)
        self.cur_batch = 0
    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    def reset(self):
        self.fname_list = self.get_lst()
        self.data_gen = self.data_iter(self.fname_list)
        self.label_gen = self.label_iter(self.fname_list)
        self.cur_batch = 0
    @property
    def provide_data(self):
        return self._provide_data
    @property
    def provide_label(self):
        return self._provide_label
    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g) for d,g in zip(self._provide_data, self.data_gen)]
            label = [mx.nd.array(g) for d,g in zip(self._provide_label, self.label_gen)]
            return mx.io.DataBatch(data, label)
        else:
            raise StopIteration
    def data_iter(self, lst):
        for i in range(self.num_sample/self.batch_size):
            batch = mx.nd.zeros((self.batch_size, 3, self.data_width, self.data_height))
            for j in range(self.batch_size):
                cur_fname = self.data_path+'/'+lst[i*self.batch_size+j]
                img = np.array(Image.open(cur_fname).resize((self.data_width,self.data_height)))
                if len(img.shape) == 2:
                    img = np.array([img,img,img])
                # batch[j] = img.reshape((3, self.data_width, self.data_height))
                tmp_img = np.swapaxes(img, 0, 2)
                if tmp_img.shape != (3,640,480):
                    tmp_img.reshape((3,640,480))
                batch[j] = tmp_img
            yield batch
    def label_iter(self, lst):
        for i in range(self.num_sample/self.batch_size):
            batch = mx.nd.zeros((self.batch_size, 1, self.label_width, self.label_height))
            for j in range(self.batch_size):
                cur_fname = self.label_path+'/'+lst[i*self.batch_size+j].split('.')[0]+'.jpeg'
                img = np.array(Image.open(cur_fname).resize((self.label_width,self.label_height)))
                img = np.array([img])
                # batch[j] = img.reshape((1, self.label_width, self.label_height))
                batch[j] = np.swapaxes(img, 2, 1)
            yield batch
    def get_lst(self):
        fname_list = []
        for _,_,f in os.walk(self.data_path):
            fname_list.extend(f)
        random.shuffle(fname_list)
        return fname_list[:SAMPLE_CNT]