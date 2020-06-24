# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:30:09 2019

@author: 81479
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from result import result_process
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

class MY_RandomForest:
    '''
    封装SVM
    '''
    def __init__(self,normalize=True,random_state = None,max_features = 'auto',
                 n_estimators = 100,criterion = 'gini',oob_score = True):
        '''
        normalize   是否在SVM过程前进行标准化
        random_state 随机种子
        max_features  每棵树的特征最大值
        n_estimators  森林中树的数量
        criterion     可选 gini entropy
        oob_score     是否选用现成的样本估计泛化精度
        
        
        
        执行完后其添加的对象有：
        self.best_param    字典形式的最佳参数
        self.svm           训练完成的SVM模型
        self.support_vec   训练得到的支持向量
        self.predict_label 预测标签
        '''
        
        self.normalize_ = normalize
        self.random_state_ = random_state
        self.max_features_ = max_features
        self.n_estimators_ = n_estimators
        self.criterion_ = criterion
        self.oob_score_ = oob_score
        self.best_param = None
        self.Scaler = None
        
    def RF_gridsearchCV(self,data,label,param_grid = None,n_splits = 5):
        
        '''
        利用网格搜索+交叉验证找最佳参数
        data        训练数据
        label       训练标签
        param_grid  网格搜索参数，字典形式
        n_splits    训练时的折数
        '''
        normalize_ = self.normalize_
        random_state_ = self.random_state_
        oob_score_ = self.oob_score_
        trainDataScale = data
        scaler = None
        if normalize_ is True:
            scaler = preprocessing.StandardScaler().fit(data)
            trainDataScale = scaler.transform(data)
        
        feature_num = trainDataScale.shape[1]
        if param_grid is None:
            param_grid = {'n_estimators': [10, 30, 60, 100,1000],
                          'criterion': ['gini','entropy'],
                          'max_features':[1,int(0.3*feature_num),int(0.5*feature_num),int(0.7*feature_num)]
                          }  #指数尺度找最佳参数
            
        kFold = StratifiedKFold(n_splits=n_splits,random_state = 777)     #设置折
        
        grid_search = GridSearchCV(RandomForestClassifier(random_state=random_state_,oob_score=oob_score_,class_weight='balanced'), 
                                   param_grid, cv=kFold,n_jobs = -1)   #网格搜索+交叉验证
        
        grid_search.fit(trainDataScale,label)   #用参数训练
        
        best_score = grid_search.best_score_   #所有折数中的最好成绩
        
        means = grid_search.cv_results_['mean_test_score']
        
        means = np.mean(means)
                
        self.mean_score = means
        
        self.Scaler = scaler
        
        self.best_param = grid_search.best_params_
        
        self.TrainDataScaler = trainDataScale
        
        return self.best_param,best_score
    
    def fit (self,train_data,train_label):
        '''
        随机森林训练过程
        如果在fit之前执行了SVM_gridsearchCV函数则不需要管max_features和n_estimators,criterion
        train_data,train_label  训练用数据以及标签
        '''
        normalize_ = self.normalize_
        random_state_ = self.random_state_
        oob_score_ = self.oob_score_
        max_features_ = self.max_features_
        n_estimators_ = self.n_estimators_
        criterion_ = self.criterion_
        trainDataScale = train_data
        
        if self.best_param is None:  #没有找最优参数
            if normalize_ is True:
                scaler = preprocessing.StandardScaler().fit(train_data)
                trainDataScale = scaler.transform(train_data)
                self.Scaler = scaler
                self.TrainDataScaler = trainDataScale
            classifier = RandomForestClassifier(random_state=random_state_,
                                                oob_score=oob_score_,
                                                max_features=max_features_,
                                                n_estimators=n_estimators_,
                                                criterion = criterion_,
                                                class_weight='balanced') 
        if self.best_param is not None:
            if normalize_ is True:
                scaler = self.Scaler
                trainDataScale = scaler.transform(train_data)
            classifier = RandomForestClassifier(**self.best_param,
                                                oob_score=oob_score_,
                                                random_state=random_state_,
                                                class_weight='balanced')
        
        classifier.fit(trainDataScale,train_label)
        
        obb_score = classifier.oob_score_
        
        self.RandomForest = classifier
        self.obb_score = obb_score
        
        return self
        
    def predict (self,test_data):
        '''
        SVM的预测过程
        test_data   需要预测的实时数据
        '''
        classifier = self.RandomForest
        normalize_ = self.normalize_
        scaler = self.Scaler
        testDataScale = test_data
        if normalize_ is True:
            testDataScale = scaler.transform(test_data)
        predict_label = classifier.predict(testDataScale)
        
        self.predict_label = predict_label
        
        return self.predict_label
    
if __name__ == '__main__':
    
    trainData = np.loadtxt('./PEMFC_Data/trainData0.35.txt')
    testData = np.loadtxt('./PEMFC_Data/testData0.65.txt')
    trainLabel = np.loadtxt('./PEMFC_Data/trainLabel0.35.txt')
    testLabelFC = np.loadtxt('./PEMFC_Data/testLabel0.65.txt')
    RF = MY_RandomForest(normalize = False,random_state = None,max_features = 'auto',
                       n_estimators = 100,criterion = 'gini',oob_score = True)
    best_para,best_score = RF.RF_gridsearchCV(trainData,trainLabel)
    '''
    best_para = {'C':1000,'gamma':10}
    best_score = 0.994785
    '''
    RF.fit(trainData,trainLabel)
    Predict_Label = RF.predict(testData)
    result_process(testLabelFC,Predict_Label,Drow = True)
      
'''
平均召回率：0.8047189891762149
各类故障的召回率：[0.80806949 0.99759036 0.21937843 0.86956522 0.7400906  0.89158644
 0.98183556 0.78277828 0.86569718 0.9955036 ]
'''    