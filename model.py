# -*- coding: utf-8 -*-
import sys, os, re

os_path = os.path.abspath('./') ; find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('demo_rows',2, 'Number of demograhic features.')
flags.DEFINE_integer('prep_rows',147, 'Number of prescribe features.')
flags.DEFINE_integer('diag_rows',49, 'Number of diagnosis features')
flags.DEFINE_integer('lab_rows',41,'Number of labtest features')
flags.DEFINE_integer('emr_rows',239,'Total number of emr rows')
flags.DEFINE_integer('months',180,'lengths of input months')
flags.DEFINE_integer('label_nums',3,'Number of label')  


class VComb_CNN(object):
    # ref : Multi-task Prediction of disease onsets from longitudinal lab tests
    # [CNN2] model structure in the paper
    def __init__(self):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.he_init = tf.contrib.layers.variance_scaling_initializer()
            with tf.variable_scope('input'):
                self.emr_input = tf.placeholder(tf.float32,shape=(None,FLAGS.emr_rows,FLAGS.months,1),name='emr')
                self.label = tf.placeholder(tf.int64,shape=(None),name='label')
                self.is_training = tf.placeholder(tf.bool, name='is_training')
            
            # vertical combinational CNN
            self.vertical_conv_1 = self._vertical_conv(self.emr_input,FLAGS.lab_rows,'vertical_conv_1')
            self.vertical_conv_2 = self._vertical_conv(self.vertical_conv_1,FLAGS.lab_rows,'vertical_conv_2')
            
            with tf.variable_scope('temporal_subnetwork'):
                # kernel_size & stride is depending on the patient's visit pattern.
                # Because the patient visits mainly weekly(7days), the stride is set on (1,7)
                self.temp_mp = tf.contrib.layers.max_pool2d(self.vertical_conv_2,
                                                            kernel_size=(1,7),stride=(1,7),scope='max_pool')
                self.temp_conv = tf.contrib.layers.conv2d(self.temp_mp,64,kernel_size=(FLAGS.lab_rows,1),
                                                          activation_fn=None,weights_initializer=self.he_init,
                                                          scope='conv')
                self.flatten=tf.contrib.layers.flatten(self.temp_conv,scope='flatten')
                tf.summary.histogram('conv_hist', self.temp_conv)

            self.fc  = self._fc(self.flatten,100,'fc1')    
            self.fc2 = self._fc(self.fc,100,'fc2')
            
            with tf.variable_scope('output'):
                self.bn = tf.contrib.layers.batch_norm(self.fc2,activation_fn=tf.nn.relu,
                                                       is_training=self.is_training,scope='batch_norm')
                self.logits = tf.layers.dense(inputs=self.bn, units=3,scope='logits')
                self.classes = tf.argmax(inputs=self.logits, axis=1,scope='classes')
                self.prob = tf.nn.softmax(self.logits, name='softmax_tensor')

            with tf.variable_scope('loss'):
                onehot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=3)
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
            tf.summary.scalar('losses', self.loss)
            
            with tf.variable_scope('accuracy'):
                self.accuracy =  tf.metrics.accuracy(labels=self.labels, predictions=self.classes)
            tf.summary.scalar('acc',self.accuracy)

    def _vertical_conv(self,input_tensor,num_outputs,layer_name):
        with tf.variable_scope(layer_name):
            vc_conv = tf.contrib.layers.conv2d(input_tensor,num_outputs,
                                     kernel_size=(FLAGS.emr_rows,1),activation_fn=None,
                                     weights_initializer=self.he_init,scope="vertical_conv")
            bn = tf.contrib.layers.batch_norm(vc_conv,activation_fn=tf.nn.relu,
                                              is_training=self.is_training,scope='batch_norm')
            tf.summary.histogram('vc_conv_hist', vc_conv)
        return bn
    
    def _fc(self,input_tensor,num_nodes,layer_name):
        with tf.variable_scope(layer_name):
            dropout = tf.contrib.layers.dropout(input_tensor,keep_prob=0.8,
                                      is_training=self.is_training,scope='dropout')
            fully_connected = tf.contrib.layers.fully_connected(dropout,num_nodes,
                                             weights_initializer=self.he_init,scope='fully_connected')
            tf.summary.histogram('fully_connected_hist', fully_connected)
        return fully_connected