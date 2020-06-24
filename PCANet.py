# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:48:16 2019

@author: 81479
"""

import numpy as np
from result import result_process,PEMFCscore_analyse_
import SklearnKPCA_Method
import SVM_Method
import RandomTree_Method
import PCA_Method_Class
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold   #保证每折中样本比例相同

class PCANet ():
    
    def __init__(self,Normalize = True,window = 7,n_components = 4,threshold=0.85,):
        '''
        Normazile  是否进行数据标准化
        windowca   patch窗口大小
        '''
        
        self.window = window
        self.Normalize = Normalize
        self.n_components = n_components
        self.threshold = threshold
    
    def get_patch(self,sample):
        '''
        实现单个样本采集patch 功能
        sample  单个样本
        '''
        window = self.window
        Feature_Num = self.Feature_Num
        sample = sample.reshape(Feature_Num,1)
        for i in range(Feature_Num):
            temp_patch = np.zeros((window,1),'float64')
            if i < (window-1)/2:
                temp_patch[0:((window-1)//2-i)] = 0
                temp_patch[((window-1)//2-i):window] = sample[0:i+((window-1)//2)+1]
            elif i >= (Feature_Num - (window-1)//2):
                temp_patch[0:(window-(window-1)//2+(Feature_Num -1- i))] = sample[i-((window-1)//2):Feature_Num]
                temp_patch[(window-(window-1)//2+(Feature_Num -1- i)):window]
            else:
                temp_patch = sample[(i-(window-1)//2):(i+(window-1)//2)+1]
            if i == 0:
                sample_patch = temp_patch
            else :
                sample_patch = np.hstack((sample_patch,temp_patch))
                
        return sample_patch
    
    def convolution(self,X_data,vec):
        '''
        对数据矩阵进行卷积操作
        X_data    为块采集后的X矩阵
        Vec       选择进行卷积的向量
        '''
        Xconvolution_data = np.zeros((X_data.shape),'float64')
        for i in range(X_data.shape[0]):
            for temp_var in range(X_data.shape[1]):
                Xconvolution_data[i,temp_var] = X_data[i,temp_var]*vec[temp_var]
            
        return Xconvolution_data
    
    def unconvolution(self,convolution_data):
        window = self.window
        Sample_Num = self.Sample_Num
        Feature_Num = self.Feature_Num
        half_window = (window-1)//2
        Xunconvolution_data = np.zeros((Sample_Num,Feature_Num),dtype='float64')
        index = 0
        for cnt in range(Sample_Num):   #按原始样本记录恢复进度
            temp_array = [0]*(Feature_Num+2*half_window)
            temp_array = np.array(temp_array,dtype='float64')
            for patch in range(Feature_Num):  #单个样本由卷积矩阵中的Feature_Num个样本组成
                Mid_point = patch + half_window
                temp_array[(Mid_point-half_window):(Mid_point+half_window+1)] = temp_array[(Mid_point-half_window):(Mid_point+half_window+1)] + convolution_data[index,:]
                index = index +1
            Xunconvolution_data[cnt,:] =  temp_array[half_window:temp_array.shape[0]-half_window]
        
        return Xunconvolution_data
    
    def fit(self,train_data,Normalize_Layer2 = True):
        '''
        训练过程
        输入 train_data
        '''
        TrainData = train_data
        Sample_Num,Feature_Num = TrainData.shape
        Normalize = self.Normalize
        n_components = self.n_components
        threshold = self.threshold
        self.Sample_Num = Sample_Num
        self.Feature_Num = Feature_Num
        '''
        步骤1 对训练数据标准化处理
        '''
        print('step 1')
        if Normalize is True:
            scaler = preprocessing.StandardScaler().fit(TrainData)    #以转置形式训练标准化
            trainDataScale = scaler.transform(TrainData)
            TrainDataScaler = trainDataScale
            self.Scaler_Layer1 = scaler
            self.TrainDataScaler = trainDataScale
        '''
        步骤2  选择窗口大小
        步骤3  根据选择的窗口大小采集每个样本的patch
        '''
        print('step 2 and 3')
        for m in range(Sample_Num):
            sample_patch = PCANet.get_patch(self,TrainDataScaler[m,:])
            if m ==0:
                TrainDataPatch = sample_patch
            else:
                TrainDataPatch = np.hstack((TrainDataPatch,sample_patch))
        TrainDataPatch = TrainDataPatch.T
        '''
        步骤4 对大矩阵做PCA计算特征值特征向量
        目前的主元个数：4
        '''
        print('step4')
        pca_model = PCA_Method_Class.MY_PCA(n_components = n_components,threshold= threshold)
        X_val,X_vec = pca_model.fit(TrainDataPatch,UCLFlag = False)
        print(pca_model.n_components)
        n_components = pca_model.n_components
        self.n_components = n_components
        '''
        步骤5 完成第一层的卷积与反卷积
        '''
        print('step 5')
        Layer1Data_namelist = []
        Layer1CData_list = []
        for v in range(n_components):      #按照主元个数进行卷积
            Layer1ConvolutionData = PCANet.convolution(self,TrainDataPatch,X_vec[:,v])
            Layer1CData_list.append(Layer1ConvolutionData)
            Layer1UnConvolutionData = PCANet.unconvolution(self,Layer1ConvolutionData)
            Layer1Data_namelist.append(Layer1UnConvolutionData)
        '''
        步骤6  第二层PCA
        '''
        print('layer2')
        Layer2CData_list = []
        Layer2Data_namelist = []               #用于存放第二层生成的十六组数据的字典
        for Data_Num in range(n_components):    #第二层的工作量是n_components倍
            TrainData_Layer2 = Layer1Data_namelist[Data_Num]
            '''
            标准化
            '''
            if Normalize_Layer2 is True:
                scaler = preprocessing.StandardScaler().fit(TrainData_Layer2)    #以转置形式训练标准化
                trainDataScale = scaler.transform(TrainData_Layer2)
                TrainDataScaler_Layer2 = trainDataScale
                self.Scaler_Layer2 = scaler
            '''
            采集patch
            '''
            for m in range(Sample_Num):
                sample_patch = PCANet.get_patch(self,TrainDataScaler_Layer2[m,:])
                if m ==0:
                    TrainDataPatch_Layer2 = sample_patch
                else:
                    TrainDataPatch_Layer2 = np.hstack((TrainDataPatch_Layer2,sample_patch))
            TrainDataPatch_Layer2 = TrainDataPatch_Layer2.T
            '''
            做PCA计算
            '''
            pca_model_Layer2 = PCA_Method_Class.MY_PCA(n_components = n_components)
            X_val_Layer2,X_vec_Layer2 = pca_model_Layer2.fit(TrainDataPatch_Layer2,UCLFlag = False)
            '''
            卷积与反卷积
            '''
            for v_L2 in range(n_components):      #按照主元个数进行卷积
                Layer2ConvolutionData = PCANet.convolution(self,TrainDataPatch_Layer2,X_vec_Layer2[:,v])
                Layer2CData_list.append(Layer2ConvolutionData)
                Layer2UnConvolutionData = PCANet.unconvolution(self,Layer2ConvolutionData)
                Layer2Data_namelist.append(Layer2UnConvolutionData)
        '''
        第二层完成
        步骤7     将数据扩展为维度很大的数据
        '''
        print('step 7')
        for Data_splice in range(n_components**2):
                if Data_splice == 0:
                    PCANet_Data =  Layer2Data_namelist[Data_splice]
                else:
                    PCANet_Data = np.hstack((PCANet_Data,Layer2Data_namelist[Data_splice]))

        return TrainDataPatch,X_vec,Layer1Data_namelist,Layer2Data_namelist,PCANet_Data,Layer1CData_list,Layer2CData_list
        
if __name__ == '__main__':
    
    '''
    全体数据
    '''
    DataMat = np.loadtxt('./PEMFC_Data/dataMat.txt')
    DataLabelFD = np.loadtxt('./PEMFC_Data/label_FD.txt')
    DataLabelFC = np.loadtxt('./PEMFC_Data/label_FC.txt')
    
    '''
    分割为 0.35 训练 和0.65测试的数据
    '''
    testData65 = np.loadtxt('./PEMFC_Data/testData0.65.txt')
    testLabel65 = np.loadtxt('./PEMFC_Data/testLabel0.65.txt')
    trainData35 = np.loadtxt('./PEMFC_Data/trainData0.35.txt')
    trainLabel35 = np.loadtxt('./PEMFC_Data/trainLabel0.35.txt')
    
    '''
    1500正常样本 和32470其他样本
    '''
    trainData = np.loadtxt('./PEMFC_Data/trainData1500.txt')
    testData = np.loadtxt('./PEMFC_Data/testData32470.txt')
    testLabel = np.loadtxt('./PEMFC_Data/testLable32470FD.txt')
    testLabelFC = np.loadtxt('./PEMFC_Data/testLable32470FC.txt')
    testLabelFD = np.loadtxt('./PEMFC_Data/testLable32470FD.txt') 
    
#%%   获取PCANet数据
    PCAnet = PCANet(window = 5,n_components = 2,threshold = 0.95)
    '''
    0.85  对应n为2
    '''
#    TrainDataPatch,X_vec,Layer1Data_namelist,Layer2Data_namelist,PCANet_Data,Layer1CData_list,Layer2CData_list = PCAnet.fit(DataMat)
    PCANet_Data = np.loadtxt('C:/Users/81479/Desktop/PCA_Net_Data/win5n4.txt')           
#%%  SVM分类器
    print('开始SVM')
    svm = SVM_Method.MY_SVM(C = 1, gamma = 1000) 
    '''                           
    '''
    
    
    skf = StratifiedKFold(n_splits = 5,shuffle=True)
    Predict_Label = []
    Label_CV = []
    for train_index,test_index in skf.split(PCANet_Data,DataLabelFC):
        train_data,test_data = PCANet_Data[train_index],PCANet_Data[test_index]
        train_label,test_label = DataLabelFC[train_index],DataLabelFC[test_index]
        svm.fit(train_data,train_label)
        Predict = svm.predict(test_data)
        Predict_Label.append(Predict)
        Label_CV.append(test_label)
    PEMFCscore_analyse_(Predict_Label,Label_CV,model = 'CV',CV_parameter = 5)
#    PEMFCscore_analyse_(Predict,test_label,model = 'Hold-out',CV_parameter = 5)
#%% SVM网格    
    '''
    网格搜索
    w5n2  [1, 1000, 1, 10000, 1, 100000, 10, 100, 10, 1000, 10, 10000, 10, 100000, 100, 100, 100, 1000, 100, 10000, 100, 100000, 1000, 10, 1000, 100, 1000, 1000, 1000, 10000, 1000, 100000, 10000, 10, 10000, 100, 10000, 1000, 10000, 10000, 10000, 100000, 100000, 10, 100000, 100, 100000, 1000, 100000, 10000, 100000, 100000]
    w5n3  [1, 100, 1, 1000, 1, 10000, 1, 100000, 10, 10, 10, 100, 10, 1000, 10, 10000, 10, 100000, 100, 10, 100, 100, 100, 1000, 100, 10000, 100, 100000, 1000, 1, 1000, 10, 1000, 100, 1000, 1000, 1000, 10000, 1000, 100000, 10000, 1, 10000, 10, 10000, 100, 10000, 1000, 10000, 10000, 10000, 100000, 100000, 0.1, 100000, 1, 100000, 10, 100000, 100, 100000, 1000, 100000, 10000, 100000, 100000]
    w5n4  [1, 100, 1, 1000, 1, 10000, 1, 100000, 10, 10, 10, 100, 10, 1000, 10, 10000, 10, 100000, 100, 10, 100, 100, 100, 1000, 100, 10000, 100, 100000, 1000, 1, 1000, 10, 1000, 100, 1000, 1000, 1000, 10000, 1000, 100000, 10000, 1, 10000, 10, 10000, 100, 10000, 1000, 10000, 10000, 10000, 100000, 100000, 0.1, 100000, 1, 100000, 10, 100000, 100, 100000, 1000, 100000, 10000, 100000, 100000]
    '''
#    time = 0
#    best_para_99 = []
#    best_para_98 = []
#    best_para_97 = []
#    best_para_95 = []
#    best_para_90 = []
#    best_para_80 = [] 
#    
#    skf = StratifiedKFold(n_splits = 2,shuffle=True)
#    for train_index,test_index in skf.split(PCANet_Data,DataLabelFC):
#        train_data,test_data = PCANet_Data[train_index],PCANet_Data[test_index]
#        train_label,test_label = DataLabelFC[train_index],DataLabelFC[test_index]
#    for C in [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]:
#        for gamma in [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]:
#            svm = SVM_Method.MY_SVM(C = C, gamma = gamma)
#            svm.fit(train_data,train_label)
#            Predict_Label = svm.predict(test_data)
#            score,recall = result_process(test_label,Predict_Label,Drow = False)
#            FaultRecallSum = 0
#            FaultLen = len(recall)-1
#            for t in range(FaultLen):
#                FaultRecallSum += recall[t+1]
#            FaultRecall = FaultRecallSum/FaultLen
#            if recall[0]>0.99 and FaultRecall>0.99:
#                best_para_99.append(C)
#                best_para_99.append(gamma)
#            elif recall[0]>0.98 and FaultRecall>0.98:
#                best_para_98.append(C)
#                best_para_98.append(gamma)
#            elif recall[0]>0.97 and FaultRecall >0.97:
#                best_para_97.append(C)
#                best_para_97.append(gamma)   
#            elif recall[0]>0.95 and FaultRecall >0.95:
#                best_para_95.append(C)
#                best_para_95.append(gamma)
#            elif recall[0]>0.95 and FaultRecall >0.9:
#                best_para_90.append(C)
#                best_para_90.append(gamma)
#            elif recall[0]>0.95 and FaultRecall >0.8:
#                best_para_80.append(C)
#                best_para_80.append(gamma)
#            print('迭代次数：{}'.format(time))
#            time = time +1   
            
'''
开始SVM
*********************************************************************************************************
执行时间： Wed Apr  8 17:28:55 2020
*********************************************************************************************************
各分类标签所含样本个数：[2194.4  390.   328.    10.  1080.   744.   316.  1030.   359.8  341.8]
预测为各分类标签数目：[2177.8  392.8  328.6   15.2 1080.2  743.4  319.  1030.   363.   344. ]
各分类标签预测正确数目：[2173.8  390.   326.6   10.  1079.2  743.2  316.  1029.   359.8  341.8]
各分类标签预测错误数目：[20.6  0.   1.4  0.   0.8  0.8  0.   1.   0.   0. ]
*********************************************************************************************************
分类正确总数：6769.4，分类错误总数：24.6
分类成绩：0.9963790583046214
*********************************************************************************************************
准确率公式：判断为该类正确格式/判断为该类总数
对各类故障的判断准确率：[0.99816593 0.99288565 0.99393364 0.66349206 0.99907476 0.99973009
 0.99062283 0.99902913 0.9911966  0.99363313]
*********************************************************************************************************
召回率公式：该类分类正确总数/该类样本数
平均召回率：0.9963790583046214
各类故障的召回率：[0.99061263 1.         0.99573171 1.         0.99925926 0.99892473
 1.         0.99902913 1.         1.        ]
*********************************************************************************************************
误判率:0.004693687277167176
*********************************************************************************************************
'''
        

        