# -*- coding: utf-8 -*-
import sys, os, re
import datetime

os_path = os.path.abspath('./') ; find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *

import preprocess.preprocess_labtest as lab
import preprocess.preprocess_diagnosis as diag
import preprocess.preprocess_prescribe as pres
import preprocess.preprocess_label as label
import preprocess.preprocess_demographic as demo
import model 
from multiprocessing import Pool
import tensorflow as tf

import data_IO as data


if __name__=='__main__':
    Model = model.VComb_CNN()
    train(Model,counts=50)