# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:12:48 2020

@author: 81479
"""


import sys
import os
import pandas as pd
current = os.path.dirname(os.path.abspath(__file__))
libsvmPath = current + '\libsvm\libsvm-3.24\python'
sys.path.append(libsvmPath)
from svmutil import *
import numpy as np
from sklearn import preprocessing

class LibSVM:
    '''
    封装libsvm
    '''
    def __init__(self,param,normalize=True):
        '''
        normalize   是否在SVM过程前进行标准化
        param       以字符串形式输入的字典
        c g         分别代表惩罚系数与核函数gamma
        '''
        
        self.normalize = normalize
        self.param = param
        self.best_param = None
        self.Scaler = None
       

        
    def fit (self,train_data,train_label):
        '''
        SVM的训练过程
        如果在fit之前执行了SVM_gridsearchCV函数则不需要管C和gamma
        train_data,train_label  训练用数据以及标签
        '''
        normalize = self.normalize
        param = self.param
        trainDataScale = train_data
        if self.best_param is None:  #没有找最优参数
            if normalize is True:
                scaler = preprocessing.StandardScaler().fit(train_data)
                trainDataScale = scaler.transform(train_data)
                self.Scaler = scaler
                self.TrainDataScaler = trainDataScale
            Classifier = svm_train(train_label,
                                   trainDataScale,
                                   param)
        if self.best_param is not None:
            if normalize is True:
                scaler = self.Scaler
                trainDataScale = scaler.transform(train_data)
            Classifier = svm_train(train_label,
                                   trainDataScale,
                                   param)
        
        
        self.LibSvm = Classifier
        
        return self
        
    def predict (self,test_data):
        '''
        SVM的预测过程
        test_data   需要预测的实时数据
        '''
        Classifier = self.LibSvm
        normalize = self.normalize
        scaler = self.Scaler
        
        testDataScale = test_data
        if normalize is True:
            testDataScale = scaler.transform(test_data)
            
        test_label = np.ones((testDataScale.shape[0],1))    
        predict_label, p_acc, p_vals =svm_predict(test_label,
                                                  testDataScale,
                                                  Classifier)
        
        predict_label = np.array(predict_label)
        self.predict_label = predict_label
        
        return self.predict_label
    
    def getW (self):
        '''
        Returns
        -------
        W : folat64
            求距离超平面公式所需要的范数.

        '''
        Classifier = self.LibSvm
        SvDict = Classifier.get_SV()     #支持向量列表
        SvCoef = [float(x[0]) for x in Classifier.get_sv_coef()]#二次规划求解得到的最小值
        MaxKey = max([x for x in SvDict[0].keys()])    #求得支持向量的特征数
        SvCoefNd = np.array(SvCoef)
        SvNd = []
        for d in SvDict:
            tmp = []
            for i in range(1,MaxKey+1):
                if i in d:
                    tmp.append(d[i])
                else:
                    tmp.append(0)    #将SvDict 的元素列表化
            SvNd.append(tmp)
        SvNd = np.array(SvNd)
        W = np.dot(SvNd.T,SvCoefNd)
        if Classifier.label[0] == -1:
            W = -W
        return W
    
    def distance(self,test_data,AbsFlag = True):
        '''
        Parameters
        ----------
        test_data : Array
            测试数据集.

        Returns
        -------
        距离值.

        '''
        Classifier = self.LibSvm
        normalize = self.normalize
        scaler = self.Scaler
        
        testDataScale = test_data
        if normalize is True:
            testDataScale = scaler.transform(test_data)
            
        test_label = np.ones((testDataScale.shape[0],1))    
        predict_label, p_acc, decision_value =svm_predict(test_label,
                                                  testDataScale,
                                                  Classifier)
        W = self.getW()
        if AbsFlag == True:
            Distance = np.divide(abs(np.array(decision_value)),np.linalg.norm(W,ord=2))
        else:
            Distance = np.divide(np.array(decision_value),np.linalg.norm(W,ord=2))
        return Distance
        
    
    
if __name__ == '__main__':

    from sklearn.model_selection import StratifiedKFold
    import result
    
    param = '-s 0 -t 2 -d 3 -g 10 -r 0.0 -c 400 -e 0.0001 -m 200 -b 1'
    Classifier = LibSVM(param)
    
    path = r'E:\MyCodeWork\图像评价\代码\数据\0703中等数据无参考29特征带真实标签\NoReferenceFeature.csv'
    s = pd.read_csv(path,index_col=0)
    Cols = [i for i in s.columns if i not in ['labelFD','labelFC']]
    train = np.array(s[Cols].values)
    label = np.array(s['labelFD'])
    skf = StratifiedKFold(n_splits = 5,shuffle=True)
    Predict_Label = []
    Label_CV = []
    for train_index,test_index in skf.split(train,label):
        train_data,test_data = train[train_index],train[test_index]
        train_label,test_label = label[train_index],label[test_index]
        
        Classifier.fit(train_data,train_label)
        Predict = Classifier.predict(test_data)
        Predict_Label.append(Predict)
        Label_CV.append(test_label)
    result.score_analyse_(Predict_Label,Label_CV,model = 'CV',CV_parameter = 5)
    
    #%% 测试SV Coef
    LibSVM = LibSVM(param)
    LibSVM.fit(train,label)
    LibW = LibSVM.getW()
    LibSv = LibSVM.LibSvm.get_SV()
    LibCoef = LibSVM.LibSvm.get_sv_coef()