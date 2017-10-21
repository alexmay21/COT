# coding=utf-8

'''
Created on Oct 9, 2017
TensorFloe Implementation of Context-dependent Translation Network (CDTN) recommender model in:
paper name
@author: Lei Mei (lei.mei@outlook.com)
'''

import tensorflow as tf
from time import time
import numpy as np

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)

class CDTN:
    def __init__(self, sess, dims, global_mean, epoch, k, k_c, keep_prob, optimizer, verbose, learning_rate=0.001, batch_size=256, reg_lambda=0.01):
        '''

        :param sess:
        :param dims:
        :param epoch:
        :param k:
        :param k_c:
        :param keep_prob:
        :param optimizer:
        :param verbose:
        :param learning_rate:
        :param batch_size:
        :param reg_lambda:
        '''
        self.sess = sess
        self.dims = dims
        print("dims",self.dims)
        self.global_mean = global_mean
        print("global mean:",self.global_mean)
        self.num_users, self.num_items = self.dims[0], self.dims[1]
        self.epoch = epoch

        self.k = k
        self.k_c = k_c
        self.keep_prob = keep_prob
        self.optimize_method = optimizer
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        self.train_rmse_record, self.train_mae_record = [], []
        self.valid_rmse_record, self.valid_mae_record = [], []
        self.test_rmse_record, self.test_mae_record = [], []

        self.build_graph()

        print("测试，输出传入到model init中的dims信息对不对")
        print("用户数：", self.num_users)
        print("物品数：", self.num_items)

    def build_graph(self):
        '''

        :return:
        '''

        # input data
        self.u_idx = tf.placeholder(tf.int32, [None], "user_id")
        self.v_idx = tf.placeholder(tf.int32, [None], "item_id")
        self.r = tf.placeholder(tf.float32, [None], "real_rating")

        self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
        self.train_phase = tf.placeholder(tf.bool)

        # 一个user-item interaction包含了多个context
        # 待做，想想怎么组织上下文的数据

        self.globalmean = tf.constant(self.global_mean, dtype=tf.float32, name="global_mean")

        self.U = weight_variable([self.num_users, self.k], "U")
        self.V = weight_variable([self.num_items, self.k], "V")

        self.U_bias = weight_variable([self.num_users], "U_bias")
        self.V_bias = weight_variable([self.num_items], "V_bias")

        with tf.name_scope("get_latent_vector"):
            self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
            self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx)
            self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
            self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)

        with tf.name_scope("user_MLP1"):
            pass

        with tf.name_scope("user_MLP2"):
            pass

        with tf.name_scope("item_MLP1"):
            pass

        with tf.name_scope("item_MLP2"):
            pass

        with tf.name_scope("predict"):
            self.r_ = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
            # add global mean, user bias and item bias
            self.r_ = tf.add(self.r_, self.globalmean)
            self.r_ = tf.add(self.r_, self.U_bias_embed)
            self.r_ = tf.add(self.r_, self.V_bias_embed)

        with tf.name_scope("loss"):
            self.loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_))
            self.reg_u_v = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)),tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
            self.reg_bias = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_bias)),tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_bias)))
            self.squared_loss_1 = tf.add(self.loss, self.reg_u_v)
            self.squared_loss = tf.add(self.squared_loss_1, self.reg_bias)

            #tf.summary.scalar("loss", self.squared_loss)

            # 将loss记录到summary
        with tf.name_scope("optimizer"):
            #self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.squared_loss)
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.squared_loss)
            #self.train_step = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.squared_loss)

        with tf.name_scope("accuracy"):
            self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_))
            self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_)))
            tf.summary.scalar("RMSE", self.RMSE)
            tf.summary.scalar("MAE", self.MAE)


        self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver()


    def get_random_block_from_data(self, data, batch_size):
        '''
        generate a random block of training data
        :param data:
        :param batch_size:
        :return:
        '''
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}

    # 打乱数据
    def shuffle_in_unison_scary(self,a,b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def construct_feeddict(self, batch_data, batch_label):
        len(batch_label)
        u_idx = batch_data.T[0]
        v_idx = batch_data.T[1]
        #print("construct:",u_idx)
        r = batch_label
        return {self.u_idx: u_idx, self.v_idx: v_idx, self.r: r}

    def train(self, Train_data, Validation_data, Test_data, result_path='result/'):
        '''

        :return:
        '''
        # 1.首先对数据进行处理

        # 2.feed

        # 3.训练
        self.writer = tf.summary.FileWriter("./logs")
        self.writer.add_graph(self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        print("Train_data(没打乱)",Train_data)
        print("Validation_data", Validation_data)
        print("Test_data", Test_data)

        # output the initial evaluation information before the first epoch if verbose>1
        if self.verbose > 0:
            t = time()
            init_train_rmse, init_train_mae = self.evaluate(Train_data, 1)
            init_valid_rmse, init_valid_mae = self.evaluate(Validation_data, 1)
            init_test_rmse, init_test_mae = self.evaluate(Test_data, 1)
            print("Init: \t train_rmse=%.4f, train_mae=%.4f [%.1f s]" % (init_train_rmse, init_train_mae, time() - t))
            print("Init: \t validation_rmse=%.4f, validation_mae=%.4f [%.1f s]" % (init_valid_rmse, init_valid_mae, time() - t))
            print("Init: \t test_rmse=%.4f, test_mae=%.4f [%.1f s]" % (init_test_rmse, init_test_mae, time() - t))

        print("总的迭代轮数：", self.epoch)
        print("训练集大小：", len(Train_data[1]))
        print("batch_size大小", self.batch_size)
        print("batch个数：", int(len(Train_data[1]) / self.batch_size))
        for epoch in range(self.epoch):
            # record the start time of each epoch
            epoch_start_time = time()

            # 打乱数据
            self.shuffle_in_unison_scary(Train_data[0], Train_data[1])
            # print("打乱后的训练数据集",Train_data)
            # print("打乱后的训练数据集的大小：",len(Train_data[1]))
            # 这边是有错的，因为在training data里，有两列
            total_batch = int(len(Train_data[1]) / self.batch_size)
            # print("打乱后：", Train_data)
            # print("打乱后的训练数据集的大小：", len(Train_data[1]))
            tr_loss = 0

            for i in range(total_batch):
                # generate a batch
                # 得到btach_size大小的训练集
                batch_data = Train_data[0][i * self.batch_size:(i + 1) * self.batch_size];
                batch_label = Train_data[1][i * self.batch_size:(i + 1) * self.batch_size];
                # print("batch_index:",i)
                # print("得到的batch data大小：",len(batch_label))
                # print("batch_data:",batch_data)
                # print("batch_data的user idx",batch_data.T[0])
                # print("batch_data的user_idx长度：", len(batch_data.T[0]))
                # 1.feed
                feed_dict = self.construct_feeddict(batch_data, batch_label)
                # print("第",i,"个batch字典内容",feed_dict[self.u_idx])

                # 2.training
                loss, _ = self.sess.run([self.squared_loss, self.train_step], feed_dict = feed_dict)
                #self.writer.add_summary(summary_str, epoch)
                #print("第",i,"个batch的loss：",loss)

                tr_loss = tr_loss + loss

            print("-----------------------------------------------------------------")
            print("第", epoch, "个epoch的loss：", tr_loss)

            epoch_end_time = time()

            # 在每一轮epoch或者特定的epoch结束后，对训练集、验证集、测试集验证结果
            train_result_rmse, train_result_mae = self.evaluate(Train_data, epoch)
            valid_result_rmse, valid_result_mae = self.evaluate(Validation_data, epoch)
            test_result_rmse, test_result_mae = self.evaluate(Test_data, epoch)

            # add the rmse&mae result of each epoch to the record list
            self.train_rmse_record.append(train_result_rmse)
            self.train_mae_record.append(train_result_mae)
            self.valid_rmse_record.append(valid_result_rmse)
            self.valid_mae_record.append(valid_result_mae)
            self.test_rmse_record.append(test_result_rmse)
            self.test_mae_record.append(test_result_mae)

            print("第",epoch,"轮迭代后的rmse结果：")
            # output the result of each epoch/or specified epoch
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain_rmse=%.4f, train_mae=%.4f [%.1f s]"
                      % (epoch + 1, epoch_end_time - epoch_start_time, train_result_rmse, train_result_mae, time() - epoch_end_time))
                print("Epoch %d [%.1f s]\tvalidation_rmse=%.4f, validation_mae=%.4f [%.1f s]"
                      % (epoch + 1, epoch_end_time - epoch_start_time, valid_result_rmse, valid_result_mae,
                         time() - epoch_end_time))
                print("Epoch %d [%.1f s]\ttest_rmse=%.4f, test_mae=%.4f [%.1f s]"
                      % (epoch + 1, epoch_end_time - epoch_start_time, test_result_rmse, test_result_mae,
                         time() - epoch_end_time))
            # early stop by using validation set


        self.saver.save(self.sess, result_path + "/model.ckpt")

    def evaluate(self, dataset, epoch):
        '''
        evaluate the result for the input dataset
        :param dataset:
        :return:
        '''
        # 1.feed
        batch_data = dataset[0]
        batch_label = dataset[1]
        # print("batch_index:",i)
        # print("得到的batch data大小：",len(batch_label))
        # print("batch_data:",batch_data)
        # print("batch_data的user idx",batch_data.T[0])
        # print("batch_data的user_idx长度：", len(batch_data.T[0]))

        # 1.feed
        feed_dict = self.construct_feeddict(batch_data, batch_label)

        # 2.evaluate
        summary_str, rmse, mae = self.sess.run([self.merged_summary, self.RMSE, self.MAE], feed_dict=feed_dict)
        self.writer.add_summary(summary_str, epoch)
        return rmse, mae