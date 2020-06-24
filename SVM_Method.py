# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:43:50 2019

@author: 81479
"""

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
import sklearn
from pylab import mpl
import time

class MY_SVM:
    '''
    封装SVM
    '''
    def __init__(self,normalize=True,C=0.01,gamma = 0.01,kernel = 'rbf'):
        '''
        normalize   是否在SVM过程前进行标准化
        C           SVM惩罚系数
        gamma       高斯核核宽度
        
        
        执行完后其添加的对象有：
        self.best_param    字典形式的最佳参数
        self.svm           训练完成的SVM模型
        self.support_vec   训练得到的支持向量
        self.predict_label 预测标签
        '''
        
        self.normalize = normalize
        self.C = C
        self.gamma = gamma
        self.best_param = None
        self.Scaler = None
        self.kernel = kernel
        
    def SVM_gridsearchCV(self,data,label,param_grid = None,n_splits = 5):
        
        '''
        利用网格搜索+交叉验证找最佳参数
        data        训练数据
        label       训练标签
        param_grid  网格搜索参数，字典形式
        n_splits    训练时的折数
        '''
        normalize = self.normalize
        Kernel = self.kernel
        trainDataScale = data
        if normalize is True:
            scaler = preprocessing.StandardScaler().fit(data)
            trainDataScale = scaler.transform(data)
            self.Scaler = scaler
        
        if param_grid is None:
            param_grid = {'C': [0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000,10000],
                          'gamma': ['auto',0.0001,0.001, 0.01, 0.1, 1, 10, 100,1000,10000]}  #指数尺度找最佳参数
            
        kFold = StratifiedKFold(n_splits=n_splits,random_state = 777)     #设置折
        
        grid_search = GridSearchCV(svm.SVC(kernel=Kernel,decision_function_shape='ovr',class_weight='balanced'), param_grid, cv=kFold,n_jobs = -1)   #网格搜索+交叉验证
        
        grid_search.fit(trainDataScale,label)   #用参数训练
        
        best_score = grid_search.best_score_   #所有折数中的最好成绩
        
        means = grid_search.cv_results_['mean_test_score']
        
        means = np.mean(means)
                
        self.mean_score = means
        
        self.best_param = grid_search.best_params_
        
        self.TrainDataScaler = trainDataScale
        
        return grid_search.best_params_,best_score
    
    def fit (self,train_data,train_label):
        '''
        SVM的训练过程
        如果在fit之前执行了SVM_gridsearchCV函数则不需要管C和gamma
        train_data,train_label  训练用数据以及标签
        '''
        normalize = self.normalize
        C = self.C
        gamma = self.gamma
        Kernel = self.kernel
        trainDataScale = train_data
        if self.best_param is None:  #没有找最优参数
            if normalize is True:
                scaler = preprocessing.StandardScaler().fit(train_data)
                trainDataScale = scaler.transform(train_data)
                self.Scaler = scaler
                self.TrainDataScaler = trainDataScale
            classifier = svm.SVC(C=C,kernel=Kernel,gamma=gamma,decision_function_shape='ovr',class_weight='balanced') #rbf高斯核函数，linear线性核函数,线性核的效果差
        if self.best_param is not None:
            if normalize is True:
                scaler = self.Scaler
                trainDataScale = scaler.transform(train_data)
            classifier = svm.SVC(**self.best_param,kernel=Kernel,decision_function_shape='ovr',class_weight='balanced')
        
        classifier.fit(trainDataScale,train_label.ravel())
        support_vec = classifier.support_vectors_
        
        self.Svm = classifier
        self.support_vec = support_vec
        
        return self
        
    def predict (self,test_data):
        '''
        SVM的预测过程
        test_data   需要预测的实时数据
        '''
        classifier = self.Svm
        normalize = self.normalize
        scaler = self.Scaler
        testDataScale = test_data
        if normalize is True:
            testDataScale = scaler.transform(test_data)
        predict_label = classifier.predict(testDataScale)
        
        self.predict_label = predict_label
        
        return self.predict_label


        
if __name__ == '__main__':
     pass