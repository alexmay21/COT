#coding=utf-8
'''
Created on Ocy 17, 2017
Processing datasets.
@author: Lei Mei (lei.mei@outlook.com)
'''

import numpy as np

class Dataset(object):
    def __init__(self, path):
        '''

        :param path:
        '''
        self.path = path
        # load the dataset and get the dimensions
        # 得到每个维度的维度数，现在是事先处理了数据集（id都是从0开始），以后要修改代码
        self.dataset, self.dims = self.read_data(path + "/SortedFoodData.txt")

        # compute the global mean
        self.global_mean = self.get_global_mean()
        # print(self.global_mean)


        # 数据集的统计信息
        self.num_users, self.num_items = self.dims[0], self.dims[1]
        self.num_ratings = len(self.dataset[1])
        # 上下文

        # 分割数据集：85% training set | 5% validation set | 10% testing set#
        self.train_ratio = 0.85
        self.validation_ratio = 0.05
        self.testing_ratio = 0.1

        self.training_set, self.validation_set, self.testing_set = self.training_validation_testing()

    def training_validation_testing(self):
        '''
        partition the dataset into training/validation/testing set
        :return:
        '''
        training_set, training_dims = self.read_data(self.path + "/ratings_train.txt")
        validation_set, validation_dims = self.read_data(self.path + "/ratings_valid.txt")
        testing_set, testing_dims = self.read_data(self.path + "/ratings_test.txt")
        print("trainset dims:",training_dims)
        print("validset dims:",validation_dims)
        print("testset dims:",testing_dims)
        return training_set, validation_set, testing_set

    def read_data(self, file_name):
        '''

        :param file_name:
        :return:
        '''
        X = np.loadtxt(file_name, dtype=float, delimiter=',')
        ndims = X.shape[1] - 1
        Y = X.T[ndims]  # rating values
        X = np.delete(X, ndims, 1).astype(int)  # index values
        dims = [X.T[i].max() + 1 for i in range(ndims)]
        return [X, Y], dims

    def cross_validation(self):
        pass

    def get_global_mean(self):
        '''

        :return: global mean
        '''
        sum = 0
        for entry in self.dataset[1]:
            sum = sum + entry
        global_mean = sum/len(self.dataset[1])
        return  global_mean


if __name__ == '__main__':
    path = "data/"
    dataset = "FoodData"
    dataset = Dataset(path + dataset)
