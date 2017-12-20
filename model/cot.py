# coding=utf-8
import tensorflow as tf
import numpy as np
from time import time
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)

class COT:
    def __init__(self, sess, dims, num_context, global_mean, epoch, k, k_c, verbose, learning_rate, batch_size, reg_lambda,  dataset):
        self.sess = sess
        self.dims = dims
        self.global_mean = global_mean
        self.num_users, self.num_items = self.dims[0], self.dims[1]
        self.epoch = epoch

        self.dataset_name = dataset

        # statistics of context information
        self.num_context_dims = len(self.dims) - 2
        self.num_context = num_context

        # dimension of effect latent vector
        self.k = k
        self.k_c = k_c
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        self.train_loss_record = []
        self.valid_loss_record = []
        self.test_loss_record = []

        self.train_rmse_record, self.train_mae_record = [], []
        self.valid_rmse_record, self.valid_mae_record = [], []
        self.test_rmse_record, self.test_mae_record = [], []

        self.build_graph()

        print("point-wise cdtn")
        print("测试，输出传入到model init中的dims信息对不对")
        print("dims",self.dims)
        print("用户数：", self.num_users)
        print("物品数：", self.num_items)
        print("global mean:", str(self.global_mean))
        print("上下文维度：", self.num_context_dims)
        print("上下文的数量有：", self.num_context)

    def build_graph(self):
        self.u_idx = tf.placeholder(tf.int32, [None], "user_id")
        self.v_idx = tf.placeholder(tf.int32, [None], "item_id")
        self.c1_idx = tf.placeholder(tf.int32, [None], "context1_id")
        self.c2_idx = tf.placeholder(tf.int32, [None], "context2_id")
        # self.c3_idx = tf.placeholder(tf.int32, [None], "context3_id")
        # self.c4_idx = tf.placeholder(tf.int32, [None], "context4_id")

        self.r = tf.placeholder(tf.float32, [None], "real_rating")

        self.globalmean = tf.constant(self.global_mean, dtype=tf.float32, name="global_mean")

        self.U = weight_variable([self.num_users, self.k], "U")
        self.V = weight_variable([self.num_items, self.k], "V")
        self.U_bias = weight_variable([self.num_users], "U_bias")
        self.V_bias = weight_variable([self.num_items], "V_bias")
        self.C = weight_variable([self.num_context, self.k_c], "C")
        self.C_bias = weight_variable([self.num_context], "C_bias")

        self.TU = weight_variable([self.k, self.k_c, self.k], "TU")
        self.TV = weight_variable([self.k, self.k_c, self.k], "TV")

        self.w_c1 = weight_variable([1], "c1_weight")
        self.w_c2 = weight_variable([1], "c2_weight")

        with tf.name_scope("get_latent_vector"):
            self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx, name="U_embed")
            self.V_embed = tf.nn.embedding_lookup(self.V, self.v_idx, name="V_embed")
            self.C1_embed = tf.nn.embedding_lookup(self.C, self.c1_idx, name="C1_embed")
            self.C2_embed = tf.nn.embedding_lookup(self.C, self.c2_idx, name="C2_embed")
            # self.C3_embed = tf.nn.embedding_lookup(self.C, self.c3_idx)
            # self.C4_embed = tf.nn.embedding_lookup(self.C, self.c4_idx)
            self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
            self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.v_idx)
            self.C1_bias_embed = tf.nn.embedding_lookup(self.C_bias, self.c1_idx)
            self.C2_bias_embed = tf.nn.embedding_lookup(self.C_bias, self.c2_idx)
            # self.C3_bias_embed = tf.nn.embedding_lookup(self.C_bias, self.c3_idx)
            # self.C4_bias_embed = tf.nn.embedding_lookup(self.C_bias, self.c4_idx)

        with tf.name_scope("get_total_context_bias"):
            self.total_context_bias = tf.add(self.C1_bias_embed, self.C2_bias_embed)
            # self.total_context_bias = tf.add(self.total_context_bias, self.C3_bias_embed)
            # self.total_context_bias = tf.add(self.total_context_bias, self.C4_bias_embed)

        with tf.name_scope("compute_ak"):
            self.ak = tf.multiply(self.w_c1, self.C1_embed) + tf.multiply(self.w_c2, self.C2_embed)
            self.ak_shape = tf.shape(self.ak)

        with tf.name_scope("u_context_aware_vector"):
            self.M_u_k = tf.einsum('lj,ijk->ilk', self.ak, self.TU)
            self.M_u_k_shape = tf.shape(self.M_u_k)

            self.u_i_k = tf.einsum('ji,ijk->jk', self.U_embed, self.M_u_k)
            # c = tf.einsum('ijk,kl->ijl', a, b)

            # self.u_i_k = tf.matmul(self.U_embed, self.M_u_k)
            self.u_i_k_shape = tf.shape(self.u_i_k)

        with tf.name_scope("v_context_aware_vector"):
            self.M_v_k = tf.einsum('lj,ijk->ilk', self.ak, self.TV)
            self.v_i_k = tf.einsum('ji,ijk->jk', self.V_embed, self.M_v_k)


        with tf.name_scope("predict"):
            # self.r_ = tf.reduce_sum(tf.multiply(self.U_under_context_embed, self.V_under_context_embed), reduction_indices=1)

            # add global mean, user bias and item bias
            self.r_ = tf.reduce_sum(tf.multiply(self.u_i_k, self.v_i_k), reduction_indices=1)
            self.r_ = tf.add(self.r_, self.globalmean)
            self.r_ = tf.add(self.r_, self.U_bias_embed)
            self.r_ = tf.add(self.r_, self.V_bias_embed)
            self.r_ = tf.add(self.r_, self.total_context_bias, name="predicted_score")

        with tf.name_scope("loss"):
            self.loss = tf.nn.l2_loss(tf.subtract(self.r, self.r_))

            # self.reg_u = tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U))
            # self.reg_v = tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V))
            # self.reg_c = tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.C))
            # self.reg_bias = tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U_bias)) + tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V_bias)) + tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.C_bias))

            params_list = tf.trainable_variables()

            self.squared_loss = self.loss
            for i in params_list:
                self.squared_loss += self.reg_lambda * tf.nn.l2_loss(i)

            # self.squared_loss = self.squared_loss + self.reg_u + self.reg_v + self.reg_c + self.reg_bias

        with tf.name_scope("optimizer"):
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.squared_loss)

        self.saver = tf.train.Saver()


    # 打乱数据
    def shuffle_in_unison_scary(self,a,b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def construct_feeddict(self, batch_data, batch_label, phase):
        u_idx = batch_data.T[0]
        v_idx = batch_data.T[1]
        c1_idx = batch_data.T[2]
        c2_idx = batch_data.T[3]
        # c3_idx = batch_data.T[4]
        # c4_idx = batch_data.T[3]
        r = batch_label
        if phase == "train":
            return {self.u_idx: u_idx, self.v_idx: v_idx, self.r: r, self.c1_idx: c1_idx, self.c2_idx: c2_idx}
        else:
            return {self.u_idx: u_idx, self.v_idx: v_idx, self.r: r, self.c1_idx: c1_idx, self.c2_idx: c2_idx}


    def train(self, Train_data, Validation_data, Test_data, result_path='save/'):
        # self.construct_item_of_user(Train_data[0])


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
        print("原训练集大小：", len(Train_data[1]))
        print("batch_size大小", self.batch_size)
        print("batch个数：", int(len(Train_data[1]) / self.batch_size))
        for epoch in range(self.epoch):
            # record the start time of each epoch
            epoch_start_time = time()

            # 打乱数据
            self.shuffle_in_unison_scary(Train_data[0], Train_data[1])

            total_batch = int(len(Train_data[1]) / self.batch_size)
            print("total_batch",total_batch)
            tr_loss = 0
            print("train_data_length",len(Train_data[1]))
            for i in range(total_batch):
                # generate a batch
                # 得到btach_size大小的训练集
                batch_data = Train_data[0][i * self.batch_size:(i + 1) * self.batch_size]
                batch_label = Train_data[1][i * self.batch_size:(i + 1) * self.batch_size]

                # 1.feed
                feed_dict = self.construct_feeddict(batch_data, batch_label, "train")
                # print("####################")
                # print(self.sess.run(self.M_u_k_shape, feed_dict=feed_dict))
                # print(self.sess.run(self.u_i_k_shape, feed_dict=feed_dict))
                # print("####################")

                loss, _ = self.sess.run([self.squared_loss, self.train_step], feed_dict=feed_dict)
                tr_loss = tr_loss + loss
                # break
            print("-----------------------------------------------------------------")
            tr_loss = tr_loss/total_batch
            print("第", epoch+1, "个epoch的loss：", tr_loss)
            self.train_loss_record.append(tr_loss)
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
        num_example = len(dataset[1])
        batch_data = dataset[0]
        batch_label = dataset[1]

        # 1.feed
        feed_dict = self.construct_feeddict(batch_data, batch_label, "evaluate")

        # 不在计算图中计算rmse和mae，因为预测的评分有可能<0,>1
        predictions = self.sess.run(self.r_, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(batch_label, (num_example,))

        if "Frappe" in self.dataset_name:
            prediction_bounded = np.maximum(y_pred, np.ones(num_example) * -1)
            prediction_bounded = np.minimum(prediction_bounded, np.ones(num_example) * 1)
        else:
            prediction_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))
            prediction_bounded = np.minimum(prediction_bounded, np.ones(num_example) * max(y_true))

        rmse = math.sqrt(mean_squared_error(y_true, prediction_bounded))
        mae = mean_absolute_error(y_true, prediction_bounded)
        #
        # # 2.evaluate
        # summary_str, rmse, mae = self.sess.run([self.merged_summary, self.RMSE, self.MAE], feed_dict=feed_dict)
        # # self.writer.add_summary(summary_str, epoch)

        return rmse, mae
