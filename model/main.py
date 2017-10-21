'''
Tensorflow implementation of Context-dependent translation network as described in:
paper name
@author: Lei Mei (lei.mei@outlook.com)
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from time import time
import tensorflow as tf
from cdtn import CDTN
from Dataset import Dataset

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='FoodData',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=4853,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=10,
                        help='Number of hidden factors.')
    parser.add_argument('--context_hidden_factor', type=int, default=64,
                        help='Number of hidden factors of context.')
    parser.add_argument('--layers', nargs='?', default='[64]',
                        help="Size of each layer.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.5]',
                        help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Bi-Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--lamda', type=float, default=0.02,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=1,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                    help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()

class data_info:
    def __init__(self):
        self.num_users = 0
        self.num_items = 0
        self.num_ratings = 0
        self.num_context_variables = 0
        self.global_mean = 0.0
        self.dims = 0

if __name__ == '__main__':
    args = parse_args()
    # dataset|epoch|batch_size|hidden_factor|context_hidden_factor|keep_prob|lambda|lr|optimizer|verbose
    #dataset = args.dataset
    epoch = args.epoch
    batch_size = args.batch_size
    hidden_factor = args.hidden_factor
    context_hidden_factor = args.context_hidden_factor
    keep_prob = args.keep_prob
    reg_lambda = args.lamda
    learning_rate = args.lr
    optimizer = args.optimizer
    verbose = args.verbose

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    training_set, validation_set, testing_set = dataset.training_set, dataset.validation_set, dataset.testing_set

    ################# context #################

    # statistics of the dataset, including user amount|item amount|rate amount|context dimensions|context conditions|
    num_users, num_items = dataset.num_users, dataset.num_items
    num_ratings = dataset.num_ratings
    num_context_dimensions = ""
    num_context_conditions = ""
    # 数据的维度信息，现在先用这个初步传入到model中，使得计算图知道用户、物品、上下文的维度等信息。
    dims = dataset.dims
    # global mean
    global_mean = dataset.global_mean

    # # 利用结构体存储数据集的一些信息
    # data_info = data_info()
    # data_info.num_users = dataset.num_users
    # data_info.num_items = dataset.num_items
    # data_info.num_ratings = dataset.num_ratings
    # data_info.dims = dataset.dims
    # data_info.global_mean = dataset.global_mean

    # print the statistic information of current used dataset
    print("Load dataset done [%.1f s]. #user=%d, #item=%d, #ratings=%d"
          % (time() - t1, num_users, num_items, num_ratings))

    # print the information of training/validation/testing set
    print("Split dataset done. #traning=%d, #validation=%d, #testing=%d"
          % (len(training_set[1]), len(validation_set[1]), len(testing_set[1])))

    # print the args information used in our proposed model
    # dataset|epoch|batch_size|hidden_factor|context_hidden_factor|keep_prob|lambda|lr|optimizer|verbose
    if args.verbose > 0:
        print("Model parameters: dataset=%s, epoch=%d, batch_size=%d, hidden_factor=%d, context_hidden_factor=%d, lambda=%.4f, lr=%.4f, optimizer=%s"
              % (args.dataset, epoch, batch_size, hidden_factor, context_hidden_factor, reg_lambda, learning_rate, optimizer))

    # training
    start_time = time()
    with tf.Session() as sess:
        # 先用dims传入model中，以后可能要改
        model = CDTN(sess, dims, global_mean, epoch, hidden_factor, context_hidden_factor, keep_prob, optimizer, verbose, learning_rate, batch_size, reg_lambda)
        model.train(training_set, validation_set, testing_set)

        sess.close()

    end_time = time()-start_time

    # Find the best validation result across iterations
    print("-----------------------------------------------------------------")
    print("#################################################################")
    print("Traing the model done. [%.1f s]"% (end_time))

    best_valid_rmse = 0
    best_valid_mae = 0

    # print(model.train_rmse_record)

    best_valid_rmse = min(model.valid_rmse_record)
    best_valid_mae = min(model.valid_mae_record)

    best_epoch_rmse = model.valid_rmse_record.index(best_valid_rmse)
    best_epoch_mae = model.valid_mae_record.index(best_valid_mae)

    print(best_epoch_rmse)
    print(best_epoch_mae)

    print("Best RMSE Iter(validation)= %d\t train_rmse = %.4f, valid_rmse = %.4f, test_rmse = %.4f"
          % (best_epoch_rmse + 1, model.train_rmse_record[best_epoch_rmse], model.valid_rmse_record[best_epoch_rmse], model.test_rmse_record[best_epoch_rmse]))

    print("Best MAE Iter(validation)= %d\t train_mae = %.4f, valid_mae = %.4f, test_mae = %.4f"
          % (best_epoch_mae + 1, model.train_mae_record[best_epoch_mae], model.valid_mae_record[best_epoch_mae], model.test_mae_record[best_epoch_mae]))
    print("#################################################################")

    # save the result in csv
    with open('result.csv', 'a') as f:
        f.write(
            "{0},{1},{2},{3},{4},{5}\n".format(learning_rate, batch_size, reg_lambda, hidden_factor, best_valid_rmse, best_valid_mae))
    tf.reset_default_graph()
    ################# 在用网格搜索确定超参时，每个启动的模型应该都要reset graph #################