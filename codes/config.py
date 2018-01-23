import os
import random
import sys

sys.dont_write_bytecode = True

BATCH_SIZE = 8
SAMPLE_CNT = 500
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
MIT1003_PATH = '../../datasets/mit1003/ALLSTIMULI'

# DATA_PATH = '../../datasets/salicon/Trn'
# LABEL_PATH = '../../datasets/salicon/Trn_Map'
DATA_PATH = '../../datasets/salicon/Trn'
LABEL_PATH = '../../datasets/salicon/Trn_Map'
# LABEL_PATH = '../../datasets/mit1003/ALLFIXATIONMAPS'

MODEL_PREFIX = '../params/deconv'
VGG_PATH = '../params/vgg/vgg16-pre-test.params'
PARAMS_PATH = '../save/deconv-0010.params'
MODEL_FAKE_PREFIX = '../save/deconv-0010.params'

POST_FIX = '.jpeg'
