import os
import random
import sys

sys.dont_write_bytecode = True

BATCH_SIZE = 8
SAMPLE_CNT = 10000
VALID_CNT = 5000
EPOCH_CNT = 100
IN_W = 640
IN_H = 480
OU_W = 80
OU_H = 60
# DATA_PATH = 'E:\\Dataset\\SALICON\\Trn\\'
# LABEL_PATH = 'E:\\Dataset\\SALICON\\Trn_Map\\'
# MODEL_PREFIX = 'E:\\Paper\\deconv_sal\\params\\deconv'
# VGG_PATH = 'E:\\Paper\\deconv_sal\\params\\vgg\\vgg16-pre-test.params'

MIT_PATH = '../../datasets/mit300'
DATA_PATH = '../../datasets/salicon/Trn'
LABEL_PATH = '../../datasets/salicon/Trn_Map'
MODEL_PREFIX = '../params/deconv'
VGG_PATH = '../save/deconv_1.params'
PARAMS_PATH = '../save/deconv_2.params'
