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

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_string('summaries_dir',BASE_PATH+'/tf_log/',' the directory of log file')
tf.app.flags.DEFINE_integer('epochs',10, 'the number of epochs')
tf.app.flags.DEFINE_integer('batch_size',128,'the size of mini-batch')

def train(Model,counts=50):
    with Model.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        train_op = tf.train.AdamOptimizer().minimize(Model.loss)
        accuracy = Model.accuracy
        update_op_acc = Model.update_op_acc
        
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        print("start!")
        valid_x,valid_y = data.make_patient_dataset('L3042', diag_counts=None, pres_counts=None, lab_counts=counts,types='validation')
        valid_set_size = data.get_patient_dataset_size('L3042', diag_counts=None, pres_counts=None, lab_counts=counts,types='validation')    
        print("validation set size : {}".format(valid_set_size))
        print("validation okay")
        with tf.Session(config=config) as sess:
            folder_name=datetime.datetime.now().strftime("%Y%m%d_%H%M")
            print("log folder is {}".format(forlder_name))
            train_writer=tf.summary.FileWriter(FLAGS.summaries_dir+"/"+folder_name + '/train',sess.graph)
            test_writer=tf.summary.FileWriter(FLAGS.summaries_dir+"/"+folder_name + '/test')
            print("epoch start!")
            set_size = data.get_patient_dataset_size('L3042', diag_counts=None, pres_counts=None, lab_counts=counts,types='train')
            print("train set size : {}".format(set_size))
            batch_nums = set_size // FLAGS.batch_size
            valid_batch_nums = valid_set_size // FLAGS.batch_size
            
            step=0; sess.run(init_op)
            for epoch in range(FLAGS.epochs):
                train_x,train_y = data.make_patient_dataset('L3042', diag_counts=None, pres_counts=None, lab_counts=counts,types='train')                
                print("train set okay")
                for batch_no in range(batch_nums):
                    emr_batch = train_x[(batch_no*FLAGS.batch_size):((batch_no+1)*FLAGS.batch_size)]
                    y_batch   = train_y[(batch_no*FLAGS.batch_size):((batch_no+1)*FLAGS.batch_size)]
                    if step% 20 == 0:  # Record summaries and test-set accuracy
                        for valid_no in range(valid_batch_nums):
                            summary,_ = sess.run([merged,update_op_acc], feed_dict={
                                'input/emr:0'  : valid_x[valid_no*FLAGS.batch_size:(valid_no+1)*FLAGS.batch_size],
                                'input/label:0' : valid_y[valid_no*FLAGS.batch_size:(valid_no+1)*FLAGS.batch_size],
                                'input/is_training:0':False
                            })
                        accuracy_value = sess.run(accuracy)
                        test_writer.add_summary(summary, step)
                        print("accuracy : {}".format(accuracy_value))
                        test_writer.flush()
                    else:  # Record train set summaries, and train
                        summary, _ = sess.run([merged,train_op],feed_dict={
                            'input/emr:0' : emr_batch,
                            'input/label:0' : y_batch,
                            'input/is_training:0':True
                        })
                        train_writer.add_summary(summary, step)
                        train_writer.flush()
                    step=step+1
                print("{} epochs is clear!".format(epoch))


if __name__=='__main__':
    Model = model.VComb_CNN()
    train(Model,counts=50)