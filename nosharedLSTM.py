#! /usr/local/bin/python

# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from config import args
from data_provider import DataProvider

class NetWork(object):
    '''
    share-parameter LSTM network
    '''

    def __init__(self):
        self.path_summary_train = args.path_predix_summary + '/train'
        self.path_summary_test = args.path_predix_summary + '/test'
        self.path_summary_logits = args.path_predix_summary +'/logits'
        self._checkpath([self.path_summary_logits,self.path_summary_test,self.path_summary_train])

    def _checkpath(self,path_list):
        for path in path_list:
            os.makedirs(path)

    def get_init_lstm_state(self):
        return np.zeros((args.station_num, self.lstm_c_size)), np.zeros((args.station_num, self.lstm_h_size))


    def _get_fc_variable(self, weight_shape):
        d = 1.0 / np.sqrt(weight_shape[0])
        bias_shape = [weight_shape[1]]
        weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=d,dtype=tf.float32,name='weights'))
        # weight = tf.Variable(tf.random_uniform(shape=weight_shape, minval=-d, maxval=d, name='weights'))
        bias = tf.Variable(tf.random_uniform(shape=bias_shape, minval=-d, maxval=d, name='bias',dtype=tf.float32))
        tf.add_to_collection('losses',tf.nn.l2_loss(weight))
        return weight, bias

    # the inference of the network
    def inference(self):

        with tf.variable_scope('batch_lstm') as vs_batch_lstm:
            # input shape is (step_size, feature_num * station_num)
            self.input = tf.placeholder(tf.float32,shape=(None,args.feature_num * args.station_num))
            # batch_input shape is (batch_size, step_size, feature_num * station_num) where batch_size is always 1
            self.batch_input = tf.expand_dims(self.input,axis=[0])
            self.input_split_list = tf.split(self.batch_input, args.station_num,axis=2)

            self.lstm_c_size = args.lstm_num_units
            self.lstm_h_size = args.lstm_num_units

            # lstm state shape is (batch_size, lstm_c_size or lstm_h_size) where lstm_c_size or lstm_h_size is always lstm_num_units
            self.lstm_c_in = tf.placeholder(tf.float32, shape=(args.station_num, self.lstm_c_size))
            lstm_c_in_list = tf.split(self.lstm_c_in,args.station_num,axis = 0)
            self.lstm_h_in = tf.placeholder(tf.float32, shape=(args.station_num, self.lstm_h_size))
            lstm_h_in_list = tf.split(self.lstm_h_in,args.station_num,axis = 0)

            
            lstm_output_list = []
            lstm_out_c_list = []
            lstm_out_h_list = []
            for i in range(args.station_num):
                with tf.variable_scope('share_lstm'+str(i)) as vs_share_lstm:
                    # if i > 0:
                    #     tf.get_variable_scope().reuse_variables()
                    lstm_cell = rnn.BasicLSTMCell(args.lstm_num_units, state_is_tuple=True)
                    lstm_cell = rnn.DropoutWrapper(lstm_cell,
                                                input_keep_prob=args.lstm_input_prob,
                                                output_keep_prob=args.lstm_output_prob,
                                                state_keep_prob = args.lstm_state_prob)
                    lstm_state_in = rnn.LSTMStateTuple(lstm_c_in_list[i], lstm_h_in_list[i])
                    # station_i_output shape is (batch_size, step_size, lstm_num_units]
                    # station_i_state is statetuple
                    station_i_output, station_i_state = tf.nn.dynamic_rnn(
                        lstm_cell,
                        self.input_split_list[i],
                        initial_state=lstm_state_in,
                        time_major=False
                    )
                    # each station's private lstm network has one FC layer
                    # station_i_output_reshape shape is (step,lstm_num_units)
                    station_i_output_reshape = tf.reshape(station_i_output,
                                                          (-1,args.lstm_num_units))
                    lstm_out_c_list.append(tf.reshape(station_i_state.c,(1,args.lstm_num_units)))
                    lstm_out_h_list.append(tf.reshape(station_i_state.h,(1,args.lstm_num_units,)))
                    # local FC_1 layer
                    local_fc_1_w, local_fc_1_b = self._get_fc_variable((args.lstm_num_units ,
                                                                  args.local_fc_1_units_num))
                    tf.summary.histogram('local_fc_1_w',local_fc_1_w)
                    # BN and drop out
                    local_fc_1_output = tf.layers.batch_normalization(tf.matmul(station_i_output_reshape, local_fc_1_w) + local_fc_1_b)
                    local_fc_1_output = tf.nn.relu(local_fc_1_output)
                    local_fc_1_output = tf.nn.dropout(local_fc_1_output,keep_prob=args.local_fc_1_prob)

                   # local FC_2 layer
                    local_fc_2_w, local_fc_2_b = self._get_fc_variable((args.local_fc_1_units_num,args.local_fc_2_units_num))
                    tf.summary.histogram('local_fc_2_w',local_fc_2_w)
                    local_fc_2_output = tf.layers.batch_normalization(tf.matmul(local_fc_1_output,local_fc_2_w) + local_fc_2_b)
                    local_fc_2_output = tf.nn.relu(local_fc_2_output)
                    local_fc_2_output = tf.nn.dropout(local_fc_2_output,keep_prob = args.local_fc_2_prob)
                    
                    lstm_output_list.append(local_fc_2_output)

            # concat the lstm output
            # lstm_output shape is (batch , station_num *  local_fc_2_units_num)
            lstm_output = tf.concat(lstm_output_list,axis=1)
            lstm_out_c = tf.concat(lstm_out_c_list,axis = 0)
            lstm_out_h = tf.concat(lstm_out_h_list,axis = 0)
            tf.summary.histogram('concat_lstm_fc_output',lstm_output)
            with tf.variable_scope('fc_layer_1') as vs_fc1:
                w1,b1 = self._get_fc_variable((args.station_num * args.local_fc_2_units_num, args.fc_1_units_num))
                tf.summary.histogram('fc_1_w',w1)
                fc1_output = tf.matmul(lstm_output,w1) + b1
                fc1_output = tf.layers.batch_normalization(fc1_output)
            #     fc1_output = tf.nn.relu(fc1_output)
            #     fc1_output = tf.nn.dropout(fc1_output,keep_prob=args.fc1_prob)

            # with tf.variable_scope('fc_layer_2') as vs_fc2:
            #     w2,b2 = self._get_fc_variable((args.fc_1_units_num,args.station_num))
            #     tf.summary.histogram('fc_2_w',w2)
            #     fc2_output = tf.matmul(fc1_output,w2) + b2
            #     fc2_output = tf.layers.batch_normalization(fc2_output)


        return fc1_output,lstm_out_c,lstm_out_h

    def get_loss(self,logits):
        # target is each station predict pm2.5 value
        # each sample in the batch is one time step
        # target shape is (step,stationo_num)
        self.target = tf.placeholder(tf.float32,shape=(None,args.station_num))

        # loss use the MSE because the problem could be seen as regression
        loss = tf.reduce_mean(tf.pow(logits - self.target ,2))
        tf.add_to_collection('losses',loss)
        # TODO: L2 loss should be in to regulization
        # total_loss = tf.add_n(tf.get_collection('losses'),'total_loss')
        total_loss = loss
        tf.summary.scalar('total_loss',total_loss)
        return loss,total_loss


    def train(self):

        with tf.variable_scope('global_scope') as scope:
            global_step = tf.Variable(0,trainable=False)
            logits,state_c,state_h = self.inference()
            loss, total_loss = self.get_loss(logits)
            lr_decay = tf.train.exponential_decay(args.lr,global_step,args.decay_steps,args.decay_rate)
            # lr_decay = args.lr
            opt = tf.train.AdamOptimizer(learning_rate=lr_decay)

            train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope.name)

            grads_vars = opt.compute_gradients(total_loss,train_variables)
            train_op = opt.apply_gradients(grads_vars,global_step=global_step)

        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.path_summary_train)
        summary_writer_test = tf.summary.FileWriter(self.path_summary_test)


        # get data
        dp = DataProvider()
        dp.load_data()

        # start train the network
        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        for batch_i in range(args.batch_num):
            train_input, train_target = dp.next_batch()

            train_lstm_c_in,train_lstm_h_in = self.get_init_lstm_state()
            feed_dict = {
                self.input: train_input,
                self.target: train_target,
                self.lstm_c_in : train_lstm_c_in,
                self.lstm_h_in : train_lstm_h_in
            }
            _,loss_value,total_loss_value,logits_value_train,np_state_c,np_state_h,summary_val = sess.run([train_op,loss,total_loss,logits,state_c, state_h, summary],feed_dict=feed_dict)
            print('batch_i = %d , loss = %.16f total_loss = %.16f' % (batch_i, loss_value, total_loss_value))
            # print('shape c is %s, shape h is %s' % (str(np_state_c.shape),str(np_state_h.shape)))
            summary_writer.add_summary(summary_val,batch_i)
            summary_writer.flush()


            # test
            if batch_i % 5 ==0:
                test_input, test_target = dp.get_test()

                train_lstm_c_in, train_lstm_h_in = self.get_init_lstm_state()
                feed_dict = {
                    self.input: test_input,
                    self.target: test_target,
                    self.lstm_c_in: np_state_c,
                    self.lstm_h_in: np_state_h
                }
                loss_value, total_loss_value, summary_val, logits_val \
                    = sess.run([loss,total_loss, summary, logits],
                               feed_dict=feed_dict)
                print('TEST batch_i = %d , test_ loss = %.16f test_total_loss = %.16f' %(batch_i, loss_value, total_loss_value))
                summary_writer_test.add_summary(summary_val, batch_i)
                summary_writer_test.flush()

                np.save(self.path_summary_logits+'/'+str(batch_i),logits_val)
                np.save(self.path_summary_logits+'/train_'+str(batch_i),logits_value_train)



if __name__ == '__main__':
    net = NetWork()
    net.train()

















