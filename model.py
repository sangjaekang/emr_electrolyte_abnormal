# -*- coding: utf-8 -*-
import sys
import os
import re

os_path = os.path.abspath('./')
find_path = re.compile('emr_electrolyte_abnormal')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *
import tensorflow as tf


class Missing_RNN(object):
    '''
    reference :
    1) LSTM model structure, Multi-task Prediction of disease onsets from longitudinal lab tests
    2) INPUT structure, Modeling Missing data in clinical time series with RNNs
    '''

    def __init__(self, configs):
        seq_length = configs['seq_length']  # the length of time sequence
        nums_lab = configs['nums_lab']  # the number of lab test
        nums_demo = configs['nums_demo']  # the number of the demographic data
        nums_diag = configs['nums_diag']  # the number of the diagnosis data
        # the number of the prescription drugs
        nums_pres = configs['nums_pres']
        num_classes = configs['num_classes']  # the number of the output labels
        dropout_rate = configs['dropout_rate']
        nums_fc_node = configs['nums_fc_node']

        hidden1_size = nums_lab
        hidden2_size = nums_pres

        self.graph = tf.Graph()

        def lstm_cell(lstm_size):
            return tf.contrib.rnn.BasicLSTMCell(lstm_size)

        with self.graph.as_default():
            with tf.variable_scope('input'):
                self.lab_input = tf.placeholder(tf.float32, shape=(
                    None, seq_length, nums_lab), name='lab')
                self.demo_input = tf.placeholder(tf.float32, shape=(
                    None, seq_length, nums_demo), name='demo')
                self.diag_input = tf.placeholder(tf.float32, shape=(
                    None, seq_length, nums_diag), name='diag')
                self.pres_input = tf.placeholder(tf.float32, shape=(
                    None, seq_length, nums_pres), name='pres')
                self.sequence_length = tf.placeholder(
                    tf.int32, shape=(None), name='sequence_length')
                self.label = tf.placeholder(
                    tf.int64, shape=(None, seq_length), name='label')
                self.weights = tf.placeholder(
                    tf.float32, shape=(None, seq_length), name='weights')
                self.is_training = tf.placeholder(tf.bool, name='is_training')

            with tf.variable_scope('LSTM_lab'):
                lstm_labtest = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell(hidden1_size) for _ in range(2)])
                rnn_outputs_lab, _ = tf.nn.dynamic_rnn(lstm_labtest, self.lab_input,
                                                       sequence_length=self.sequence_length,
                                                       dtype=tf.float32)

            with tf.variable_scope('LSTM_prescribe'):
                lstm_prescribe = tf.contrib.rnn.MultiRNNCell(
                    [lstm_cell(hidden2_size) for _ in range(2)])
                rnn_output_pres, _ = tf.nn.dynamic_rnn(lstm_prescribe, self.pres_input,
                                                       sequence_length=self.sequence_length,
                                                       dtype=tf.float32)

            with tf.variable_scope('fully_connected'):
                Diag_reshape = tf.reshape(self.diag_input, [-1, nums_diag])
                #Demo_reshape = tf.reshape(self.demo_input, [-1, nums_demo])
                Lab_reshape = tf.reshape(rnn_outputs_lab, [-1, hidden_size])
                Pres_reshape = tf.reshape(rnn_output_pres, [-1, nums_pres])

                diag_fc1 = tf.contrib.layers.fully_connected(Diag_reshape, 10)
                diag_fc2 = tf.contrib.layers.fully_connected(
                    diag_fc1, 10, activation_fn=tf.nn.softmax)

                pres_fc1 = tf.contrib.layers.fully_connected(Pres_reshape, 10)
                pres_fc2 = tf.contrib.layers.fully_connected(
                    pres_fc1, 10, activation_fn=tf.nn.softmax)

                concat_input = tf.concat(
                    [Lab_reshape, diag_fc2, pres_fc2], axis=1)
                dropout1 = tf.contrib.layers.dropout(concat_input, keep_prob=dropout_rate,
                                                     is_training=self.is_training, scope='dropout1')
                fully_connected1 = tf.contrib.layers.fully_connected(
                    dropout1, nums_fc_node, scope='fc1')
                dropout2 = tf.contrib.layers.dropout(fully_connected1, keep_prob=dropout_rate,
                                                     is_training=self.is_training, scope='dropout2')
                fully_connected2 = tf.contrib.layers.fully_connected(
                    dropout2, nums_fc_node, scope='fc2')
                bn = tf.contrib.layers.batch_norm(fully_connected2,
                                                  is_training=self.is_training, scope="batch_norm")

            with tf.variable_scope('output'):
                dense = tf.layers.dense(
                    inputs=bn, units=num_classes, name='logits')
                self.logits = tf.reshape(dense, [-1, seq_length, num_classes])
                self.prediction = tf.argmax(self.logits, axis=2, name='label')
                self.prob = tf.nn.softmax(self.logits, name='prob')

            with tf.variable_scope('loss'):
                weights = tf.ones_like(self.label, dtype=tf.float32)
                sequence_loss = tf.contrib.seq2seq.sequence_loss(
                    logits=self.logits, targets=self.label, weights=weights)
                self.loss = tf.reduce_mean(sequence_loss)
            tf.summary.scalar('LOSS', self.loss)

            with tf.variable_scope('accuracy'):
                correct_prediction = tf.equal(self.prediction, self.label)
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('ACCURACY', self.accuracy)


# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('demo_rows',2, 'Number of demograhic features.')
# flags.DEFINE_integer('prep_rows',147, 'Number of prescribe features.')
# flags.DEFINE_integer('diag_rows',49, 'Number of diagnosis features')
# flags.DEFINE_integer('lab_rows',41,'Number of labtest features')
# flags.DEFINE_integer('emr_rows',239,'Total number of emr rows')
# flags.DEFINE_integer('months',180,'lengths of input months')
# flags.DEFINE_integer('label_nums',3,'Number of label')

# class VComb_CNN(object):
#     # ref : Multi-task Prediction of disease onsets from longitudinal lab tests
#     # [CNN2] model structure in the paper
#     def __init__(self):
#         #tf.reset_default_graph()
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             self.he_init = tf.contrib.layers.variance_scaling_initializer()
#             with tf.variable_scope('input'):
#                 self.emr_input = tf.placeholder(tf.float32,shape=(None,FLAGS.emr_rows,FLAGS.months,1),name='emr')
#                 self.label = tf.placeholder(tf.int64,shape=(None),name='label')
#                 self.is_training = tf.placeholder(tf.bool, name='is_training')

#             # vertical combinational CNN
#             self.vertical_conv_1 = self._vertical_conv(self.emr_input,FLAGS.lab_rows,'vertical_conv_1')
#             self.vertical_conv_2 = self._vertical_conv(self.vertical_conv_1,FLAGS.lab_rows,'vertical_conv_2')

#             with tf.variable_scope('temporal_subnetwork'):
#                 # kernel_size & stride is depending on the patient's visit pattern.
#                 # Because the patient visits mainly weekly(7days), the stride is set on (1,7)
#                 self.temp_mp = tf.contrib.layers.max_pool2d(self.vertical_conv_2,
#                                                             kernel_size=(1,7),stride=(1,7),scope='max_pool')
#                 self.temp_conv = tf.contrib.layers.conv2d(self.temp_mp,64,kernel_size=(FLAGS.lab_rows,1),
#                                                           activation_fn=None,weights_initializer=self.he_init,
#                                                           scope='conv')
#                 self.flatten=tf.contrib.layers.flatten(self.temp_conv,scope='flatten')
#                 tf.summary.histogram('conv_hist', self.temp_conv)

#             self.fc  = self._fc(self.flatten,100,'fc1')
#             self.fc2 = self._fc(self.fc,100,'fc2')

#             with tf.variable_scope('output'):
#                 self.bn = tf.contrib.layers.batch_norm(self.fc2,activation_fn=tf.nn.relu,
#                                                        is_training=self.is_training,scope='batch_norm')
#                 self.logits = tf.layers.dense(inputs=self.bn, units=3,name='logits')
#                 self.classes = tf.argmax(self.logits, axis=1,name='classes')
#                 self.prob = tf.nn.softmax(self.logits, name='softmax_tensor')

#             with tf.variable_scope('loss'):
#                 onehot_labels = tf.one_hot(indices=tf.cast(self.label, tf.int32), depth=3)
#                 self.loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.logits)
#             tf.summary.scalar('losses', self.loss)

#             with tf.variable_scope('accuracy'):
#                 correct_prediction = tf.equal(self.classes,self.label)
#                 self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#             tf.summary.scalar('acc',self.accuracy)

#     def _vertical_conv(self,input_tensor,num_outputs,layer_name):
#         with tf.variable_scope(layer_name):
#             vc_conv = tf.contrib.layers.conv2d(input_tensor,num_outputs,
#                                      kernel_size=(FLAGS.emr_rows,1),activation_fn=None,
#                                      weights_initializer=self.he_init,scope="vertical_conv")
#             bn = tf.contrib.layers.batch_norm(vc_conv,activation_fn=tf.nn.relu,
#                                               is_training=self.is_training,scope='batch_norm')
#             tf.summary.histogram('vc_conv_hist', vc_conv)
#         return bn

#     def _fc(self,input_tensor,num_nodes,layer_name):
#         with tf.variable_scope(layer_name):
#             dropout = tf.contrib.layers.dropout(input_tensor,keep_prob=0.8,
#                                       is_training=self.is_training,scope='dropout')
#             fully_connected = tf.contrib.layers.fully_connected(dropout,num_nodes,
#                                              weights_initializer=self.he_init,scope='fully_connected')
#             tf.summary.histogram('fully_connected_hist', fully_connected)
#         return fully_connected
