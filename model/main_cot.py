import argparse
from time import time
import tensorflow as tf
from cot import COT
from Dataset import Dataset
import os

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run.")
    parser.add_argument('--algorithm', nargs='?', default='cdtn',
                        help='Choose which algorith to run. mf|cdtn')
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='FoodData/transfer',
                        help='Choose a dataset.')
    parser.add_argument('--grid_search', type=int, default=0,
                        help='Whether to perform grid search.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', nargs='?', default='[128,256,512,1024]',
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--context_hidden_factor', nargs='?', default='[32,64,128,256]',
                        help='Number of hidden factors of context.')
    parser.add_argument('--effect_hidden_factor', nargs='?', default='[32,64,128,256]',
                        help='Number of hidden factors of effect.')
    parser.add_argument('--mlp1_layers', nargs='?', default='[0,1,2,3,4,5,6,7,8]',
                        help="Number of layer in MLP1.")
    parser.add_argument('--mlp2_layers', nargs='?', default='[0,1,2,3,4,5,6,7,8]',
                        help="Number of layer in MLP2.")
    parser.add_argument('--mlp1_hidden_size', nargs='?', default='[128,256]',
                        help="Dimension of the hidden size of mlp1.")
    parser.add_argument('--mlp2_hidden_size', nargs='?', default='[128,256]',
                        help="Dimension of the hidden size of mlp2.")
    parser.add_argument('--keep_prob', nargs='?', default='[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]',
                        help='Keep probabilityout.')
    parser.add_argument('--lamda', nargs='?', default='[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0]',
                        help='Regularizer for parameters')
    parser.add_argument('--lr', nargs='?', default='[0.001,0.005,0.01,0.05,0.1]', help='Learning rate.')
    parser.add_argument('--predict_way', nargs='?', default='[-1,0,1]',
                        help='-1:r=U_under_context*V, 0:r=U*V_under_context, 1:r=U_under_context*V_under_context')
    parser.add_argument('--last_layer_activation', nargs='?', default='[0,1]',
                        help='Whether to use activation function in the last layer of MLP1 and MLP2.')
    parser.add_argument('--bi_linear', nargs='?', default='[0,1]',
                        help='Whether to add bi-linear layer upon the input layer of MLP1 and MLP2.')
    parser.add_argument('--bi_linear_mlp1_hidden_size', nargs='?', default='[64,128,256]',
                        help='Hidden size of bi-linear of MLP1.')
    parser.add_argument('--bi_linear_mlp2_hidden_size', nargs='?', default='[64,128,256]',
                        help='Hidden size of bi-linear of MLP2.')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
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

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    args = parse_args()

    # config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # algorithm|dataset|epoch|batch_size|hidden_factor|context_hidden_factor|effect_hidden_factor|keep_prob|lambda|lr|optimizer|verbose
    algorithm = args.algorithm
    dataset = args.dataset
    path = args.path
    epoch = args.epoch
    grid_search = args.grid_search
    verbose = args.verbose
    # hidden_factor = args.hidden_factor

    hidden_factor = 8
    batch_size = 128
    context_hidden_factor = 4
    reg_lambda = 0.001
    learning_rate = 0.001

    # Loading data
    t1 = time()
    dataset = Dataset(path + dataset)
    training_set, validation_set, testing_set = dataset.training_set, dataset.validation_set, dataset.testing_set

    # statistics of the dataset, including user amount|item amount|rate amount|context dimensions|context conditions|
    num_users = dataset.num_users
    num_items = dataset.num_items
    num_ratings = dataset.num_ratings
    dims = dataset.dims
    num_context_dims = len(dims) - 2
    num_context = dataset.num_context
    # global mean
    global_mean = dataset.global_mean

    print("=============================================================================")
    # print the statistic information of current used dataset
    print("Load dataset done [%.1f s]. #user=%d, #item=%d, #ratings=%d"
          % (time() - t1, num_users, num_items, num_ratings))

    # print the information of training/validation/testing set
    print("Split dataset done. #traning=%d, #validation=%d, #testing=%d"
          % (len(training_set[1]), len(validation_set[1]), len(testing_set[1])))

    # print context information
    print("num_users:", num_users)
    print("num_items:", num_items)
    print("num_interactions:", num_ratings)
    print("num_context_dims:",num_context_dims)
    print("num_context:", num_context)
    print("dims:", dims)

    print("=============================================================================")

    # training
    start_time = time()
    with tf.Session(config=config) as sess:

        model = COT(sess, dims, num_context, global_mean, epoch, hidden_factor, context_hidden_factor, verbose, learning_rate, batch_size, reg_lambda, args.dataset)
        model.train(training_set, validation_set, testing_set)

        print("#################################################################")
        print("CDTN")
        print("输出训练过程的loss等信息：")
        print("training loss")
        print(model.train_loss_record)
        print("training rmse")
        print(model.train_rmse_record)
        print("valid rmse")
        print(model.valid_rmse_record)
        print("testing rmse")
        print(model.test_rmse_record)
        print("Traing the model done. [%.1f s]" % (time() - start_time))

        best_valid_rmse = 0
        best_valid_mae = 0

        best_valid_rmse = min(model.valid_rmse_record)
        best_valid_mae = min(model.valid_mae_record)

        best_epoch_rmse = model.valid_rmse_record.index(best_valid_rmse)
        best_epoch_mae = model.valid_mae_record.index(best_valid_mae)

        # print(best_epoch_rmse)
        # print(best_epoch_mae)

        print("Best RMSE Iter(validation)= %d\t train_rmse = %.4f, valid_rmse = %.4f, test_rmse = %.4f"
              % (best_epoch_rmse + 1, model.train_rmse_record[best_epoch_rmse],
                 model.valid_rmse_record[best_epoch_rmse], model.test_rmse_record[best_epoch_rmse]))

        print("Best MAE Iter(validation)= %d\t train_mae = %.4f, valid_mae = %.4f, test_mae = %.4f"
              % (best_epoch_mae + 1, model.train_mae_record[best_epoch_mae],
                 model.valid_mae_record[best_epoch_mae], model.test_mae_record[best_epoch_mae]))
        sess.close()
