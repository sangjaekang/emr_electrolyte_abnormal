# -*- coding: utf-8 -*-
import sys, os, re

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

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('summaries_dir',BASE_PATH+'/tf_log/',' the directory of log file')
tf.app.flags.DEFINE_integer('epochs',10, 'the number of epochs')
tf.app.flags.DEFINE_integer('batch_size',128,'the size of mini-batch')

with Model.graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    train_op = tf.train.AdamOptimizer().minimize(Model.loss)
    accuracy = Model.accuracy
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    print("start!")
    valid_x,valid_y = data.make_patient_dataset('L3042', diag_counts=None, pres_counts=None, lab_counts=50,types='validation')
    print("validation set size : {}".format(valid_x.shape))
    print("validation okay")

    with tf.Session(config=config) as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
        set_size = data.get_patient_dataset_size('L3042', diag_counts=None, pres_counts=None, lab_counts=50,types='train')
        valid_set_size = data.get_patient_dataset_size('L3042', diag_counts=None, pres_counts=None, lab_counts=50,types='train')    
        batch_nums = set_size // FLAGS.batch_size
        valid_batch_nums = valid_set_size // FLAGS.batch_size
        step=0
        print("epoch start!")
        for epoch in range(FLAGS.epochs):
            train_x,train_y = data.make_patient_dataset('L3042', diag_counts=None, pres_counts=None, lab_counts=50,types='train')    
            print("train set size : {}".format(train_x.shape))
            print("train set okay")
            for batch_no in range(batch_nums):
                emr_batch = train_x[(batch_no*FLAGS.batch_size):((batch_no+1)*FLAGS.batch_size)]
                y_batch   = train_y[(batch_no*FLAGS.batch_size):((batch_no+1)*FLAGS.batch_size)]
            
                if step% 100 == 0:  # Record summaries and test-set accuracy
                    acc_list = []
                    for valid_no in range(valid_batch_nums):
                        summary, acc = sess.run([merged, accuracy], feed_dict={
                            'input/emr:0'  : valid_x[valid_no*FLAGS.batch_size:(valid_no+1)*FLAGS.batch_size],
                            'input/label:0' : valid_y[valid_no*FLAGS.batch_size:(valid_no+1)*FLAGS.batch_size],
                            'input/is_training:0':False
                        })
                        test_writer.add_summary(summary, step)
                        test_writer.flush()
                        acc_list.append(float(acc))
                    print('Accuracy at step %s: %s' % (step, np.mean(acc_list)))
                else:  # Record train set summaries, and train
                    summary, _ = sess.run([merged,train_op],fead_dict={
                        'input/emr:0' : emr_batch,
                        'input/label:0' : y_batch,
                        'input/is_training:0':True
                    })
                    train_writer.add_summary(summary, step)
                    train_writer.flush()
                step=step+1
            print("{} epochs is clear!".format(epoch))