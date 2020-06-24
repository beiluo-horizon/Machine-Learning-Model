# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:25:43 2019

@author: 81479
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import KernelPCA
import math
import colorsys
import random
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from result import TEscore_analyse_
from Ucl import UCL
from sklearn.model_selection import StratifiedKFold   #保证每折中样本比例相同
from matplotlib.font_manager import FontProperties
from pylab import mpl
import time


class MY_KPCA:
    def __init__(self,n_components=None,normalize=True,threshold=0.85,
                 kernel = 'rbf',gamma = 0.1,fit_inverse_transform = False,
                 alpha=0.01,algorithm='kde'):
        '''
        初始化参数
        n_components    保留特征个数 如果为None,则使用threshold
        normalize       是否在运算前标准化数据
        threshold       方差阈值
        kernel          核函数总类
        gamma           核函数宽度
        fit_inverse_transform     是否拟合恢复原数据
        alpha           求分布时的参数
        algorithm       求控制限的算法
        '''
        
        self.n_components = n_components
        self.normalize = normalize
        self.threshold = threshold
        self.kernel = kernel
        self.gamma = gamma
        self.fit_inverse_transform = fit_inverse_transform
        self.alpha = alpha
        self.algorithm = algorithm
        
    @staticmethod   
    def Normalize_K(K):
        '''
        实现核函数矩阵的标准化
        公式K_ = K-KE-EK+EKE
        其中E是元素为1/n的nxn常数矩阵
        '''
        N = K.shape[0]
        E = np.ones((N, N))/N
        K_ = K - np.dot(K,E) - np.dot(E,K) + np.dot(np.dot(E,K),E)
        
        return K_
    @staticmethod
    def Normalize_vec(vec):
        '''
        实现特征向量的标准化
        公式：V/||V||
        '''
        for i in range(vec.shape[1]):
            sum_lenth = 0
            for j in range(vec.shape[0]):
                sum_lenth = sum_lenth + vec[i,j]**2
            lenth = math.sqrt(sum_lenth)
            vec[:,i] = 1/lenth*vec[:,i]
        return vec     
    
    def Get_statics(self,testdata,model = '1'):
        '''
        获取新样本的统计量
        输入新样本数据
        model代表了两种不同的SPE计算方式
        model 1 文章：
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7105134&tag=1
        model 2 文章：
        https://pdf.sciencedirectassets.com/271351/1-s2.0-S0169743900X01578/1-s2.0-S0169743904001224/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCICIi72hv%2BZHxueRVnfq9RdQIIPuTDf8JgtC9mlvhuJD8AiBjvCTtjHQct0DUJ1BfnDIOKUc2bDwK%2F80EfYNzWKEGLCrPAggiEAIaDDA1OTAwMzU0Njg2NSIMxSr59t3Z6KhgHjdEKqwCb042bOisiDNy0L3ROxP%2FWjclO0ED5Of6MsviAj5JMQFBzA%2FsN96BzIAcQEmkf9t%2BFlixIZOqTo9hBueGCtR6Nnsr%2FVN7ty6oDRIcSj1Ye3MWEOpDa8HLLVlqP4cTUbAsOz1VN%2BLsr6dzWY9vUs2%2FbSX2pmg%2FSbrFPT6mvkK3W3aGmXU81jjT5LuFXY9lAsBiq2ky9O0gXeW9MoSkiLsMqMtgM1urJ%2Ff%2BYZ5k%2FHExnd3Ceo1Nrsg%2FZ4IFrDWUbDg9pFoxV09LDSqpXYQN6hzldeiSS9v22X%2Fw1ZvOs3m6PAorYt1Z5nFKOiwHDJrO3llSIeuHcouT2erJUmpv7xwK%2BffvZgBM8JLujFz%2BOAuT9gn8uPB5oENS7KyRFkrYCAFFtqugWJzf4EgLpOV%2BMN7Ky%2B8FOtECr8PZ6tnJwa%2BvdjLlzi65jNEpqhdqBJchKepKWnM4hbNGnaEs76%2BJhVxhMTFCSo3Hz1je8W7NMTeqeoKfZvocAxZeNR1nkqbLQsL%2BSPWIKj85ln%2F%2FTMqB43bdPcrZn%2BEcTPsC6scLbl1ZvUsJddatT%2Bda0WwsQyrvNGnWi0fJofF8eWt5JpOtIALyTbqBWeGtqzCAl6lYm0xNGNfEliKFbmPFPyQ3%2BG0vY5PMpxXbFbbwFD9dOAbs2Ns2psQdPKosZAX2ejAicr8PGUS%2F%2FZZPM2WdTkjQSA9gsxag8cvq2eKHrO8g4%2FnvwX7uy9DpV%2B%2BimAxyQo3i7UXzHUHdPRFPtM8PeL0gSOXiNSvaEZItz1Nf5mshRx51o1FwCeOr%2B0W9RYS%2F64jB9on8HSqYFg73Ub8pofdv20AQeKjIhI64dvwTu5biGu%2BKfGBO%2FX10HN%2FD2g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191213T023933Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQJ2RI24N%2F20191213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=63b02081dad7309a8555ca8e6309e041988f3293937bcbe380ac365a5e7e5ee9&hash=bba449bb44df4f0e0ca1c9cf6aa39ee7d890d5bd1b929657eccf2e7de0783783&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0169743904001224&tid=spdf-847c11cf-84a3-44a8-868f-91be0b4ad0cd&sid=87b762f85af5c642fa89fb7538d7b44c5c6agxrqa&type=client
        model = 1   公式：  SPE = ∑(j,n)tj^2 - ∑(j,k)tj^2 
        model = 2   公式：  SPE = K(x_new,x_new)-t^T*t
        
        '''
        T = MY_KPCA.transform(self,testdata,Flag = 'N')
        total_T = MY_KPCA.transform(self,testdata,Flag = 'Total')
        T = T.T
        total_T = total_T.T
        T2 = [0]*testdata.shape[0]
        SPE = [0]*testdata.shape[0]
        
        for i in range(testdata.shape[0]):
            T2[i] = np.dot(np.dot(T[:,i].T,self.lamda),T[:,i])
            
        for i in range(testdata.shape[0]):
            if model =='1':
                temp_T = total_T[:,i]
                sum_all = 0
                sum_k = 0 
                for t in range(len(temp_T)):
                    sum_all = sum_all + temp_T[t]**2
                    if t <self.n_components:
                        sum_k = sum_k + temp_T[t]**2
                SPE[i] = sum_all - sum_k
            if model == '2':
                SPE[i] = 1 - np.dot(T[:,i].T,T[:,i]) 
                
        return T2,SPE
    
    def Get_Ucls(self,model = '1'):
        '''
        计算控制限
        T2控制限与PCA相同，差别是n^2
        model代表了两种不同的SPE控制限计算方式
        model 1 文章：
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7105134&tag=1
        model 2 文章：
        https://pdf.sciencedirectassets.com/271351/1-s2.0-S0169743900X01578/1-s2.0-S0169743904001224/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMr%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCICIi72hv%2BZHxueRVnfq9RdQIIPuTDf8JgtC9mlvhuJD8AiBjvCTtjHQct0DUJ1BfnDIOKUc2bDwK%2F80EfYNzWKEGLCrPAggiEAIaDDA1OTAwMzU0Njg2NSIMxSr59t3Z6KhgHjdEKqwCb042bOisiDNy0L3ROxP%2FWjclO0ED5Of6MsviAj5JMQFBzA%2FsN96BzIAcQEmkf9t%2BFlixIZOqTo9hBueGCtR6Nnsr%2FVN7ty6oDRIcSj1Ye3MWEOpDa8HLLVlqP4cTUbAsOz1VN%2BLsr6dzWY9vUs2%2FbSX2pmg%2FSbrFPT6mvkK3W3aGmXU81jjT5LuFXY9lAsBiq2ky9O0gXeW9MoSkiLsMqMtgM1urJ%2Ff%2BYZ5k%2FHExnd3Ceo1Nrsg%2FZ4IFrDWUbDg9pFoxV09LDSqpXYQN6hzldeiSS9v22X%2Fw1ZvOs3m6PAorYt1Z5nFKOiwHDJrO3llSIeuHcouT2erJUmpv7xwK%2BffvZgBM8JLujFz%2BOAuT9gn8uPB5oENS7KyRFkrYCAFFtqugWJzf4EgLpOV%2BMN7Ky%2B8FOtECr8PZ6tnJwa%2BvdjLlzi65jNEpqhdqBJchKepKWnM4hbNGnaEs76%2BJhVxhMTFCSo3Hz1je8W7NMTeqeoKfZvocAxZeNR1nkqbLQsL%2BSPWIKj85ln%2F%2FTMqB43bdPcrZn%2BEcTPsC6scLbl1ZvUsJddatT%2Bda0WwsQyrvNGnWi0fJofF8eWt5JpOtIALyTbqBWeGtqzCAl6lYm0xNGNfEliKFbmPFPyQ3%2BG0vY5PMpxXbFbbwFD9dOAbs2Ns2psQdPKosZAX2ejAicr8PGUS%2F%2FZZPM2WdTkjQSA9gsxag8cvq2eKHrO8g4%2FnvwX7uy9DpV%2B%2BimAxyQo3i7UXzHUHdPRFPtM8PeL0gSOXiNSvaEZItz1Nf5mshRx51o1FwCeOr%2B0W9RYS%2F64jB9on8HSqYFg73Ub8pofdv20AQeKjIhI64dvwTu5biGu%2BKfGBO%2FX10HN%2FD2g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20191213T023933Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQJ2RI24N%2F20191213%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=63b02081dad7309a8555ca8e6309e041988f3293937bcbe380ac365a5e7e5ee9&hash=bba449bb44df4f0e0ca1c9cf6aa39ee7d890d5bd1b929657eccf2e7de0783783&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0169743904001224&tid=spdf-847c11cf-84a3-44a8-868f-91be0b4ad0cd&sid=87b762f85af5c642fa89fb7538d7b44c5c6agxrqa&type=client
        model = 1   公式：  
        SPELIMIT = gX^2h
        其中g = b/2a   h=2a^2/b     a是正常样本SPE值的平均值  b是正常样本SPE值的方差
        model = 2   与PCA控制限相同
        '''
        alpha = self.alpha
        n = self.samplesNum
        k = self.n_components
        algorithm = self.algorithm
        
        if algorithm == 'formula':
            T2UCL = k* (n -1)/(n*(n-k)) * stats.f.isf(alpha,k,n-k)     #论文中的公式
#           T2UCL =  k * (n*n -1)/(n*(n-k)) * stats.f.isf(alpha,k,n-k)   #PCA的公式
        
            if model == '1':
                T2_normal,SPE_normal = MY_KPCA.Get_statics(self,self.traindata,model = '1')
                SPE_array=np.array(SPE_normal)
                a = np.mean(SPE_array)
                b = np.var(SPE_array)
                g = b/(2*a)
                h = 2*a**2/b
                SPEUCL = g*stats.chi2.ppf(alpha, h)
            
            if model == '2':
                ca = stats.norm.isf(alpha)
                # k之后的特征值
                theta1 = np.power(self.val[k:],1.0).sum()
                theta2 = np.power(self.val[k:],2.0).sum()
                theta3 = np.power(self.val[k:],3.0).sum()
                h0 = 1- 2*theta1*theta3/(3*theta2*theta2)
                SPEUCL = theta1*np.power(ca*h0*np.sqrt(2*theta2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1*theta1) ,1.0/h0)
        elif algorithm == 'kde':
            
            T2_normal,SPE_normal = MY_KPCA.Get_statics(self,self.traindata,model = model)
            
            T2_normal = np.array(T2_normal)
            SPE_normal = np.array(SPE_normal)
            kde = UCL.searchKde(T2_normal, auto= True)
            T2UCL = UCL.KDE(kde,auto=True,data=T2_normal)
            kde = UCL.searchKde(SPE_normal, auto= True)
            SPEUCL = UCL.KDE(kde,auto=True,data=SPE_normal)
        
        return T2UCL,SPEUCL
    
    @staticmethod
    def calComponentSelectNums(val,threshold=0.85):
        """
        计算特征值占比达到 threshold 时的特征值数目。
        Parameters
        ----------
        val : array-like
            存放特征值的一维数组或列表，传递时可不用从大到小排序。
        threshold : float , 默认为 0.85
            int:  自动确定特征向量个数时的阈值

        Returns
        -------
        nums : 大于阈值时的特征值的数目
        """
        rates = -np.sort(-val) / val.sum()
        rateSum = 0.0
        nums = len(rates)
        for i in range(nums):
            rateSum += rates[i]
            if rateSum >= threshold:
                nums = i + 1
                break
        return nums 
    
    def fit(self,train_data):
        '''
        KPCA模型的训练过程
        
        train_data  训练数据  会根据normalize做标准化
        '''
        n_components = self.n_components
        normalize = self.normalize
        threshold = self.threshold
        kernel = self.kernel
        gamma = self.gamma
        fit_inverse_transform = self.fit_inverse_transform
        samplesNum,featuresNum = train_data.shape
        TrainDataScaler = None
        if normalize is True:
            scaler = preprocessing.StandardScaler().fit(train_data)
            trainDataScale = scaler.transform(train_data)
            TrainDataScaler = trainDataScale
            self.Scaler = scaler
            self.TrainDataScaler = trainDataScale
        else:
            TrainDataScaler = train_data
            
        kpca_model = KernelPCA(kernel = kernel,
                               fit_inverse_transform=fit_inverse_transform,
                               gamma = gamma,
                               )
        kpca_model.fit(TrainDataScaler)    #用训练数据训练模型
        
        val = kpca_model.lambdas_         #核矩阵的特征值与特征向量
        vec = kpca_model.alphas_
        kernel_matrix = kpca_model._get_kernel(TrainDataScaler)
        
        self.val = val
        self.vec = vec
        
        if n_components is None:
            n_components = MY_KPCA.calComponentSelectNums(val,threshold=threshold)
        
        self.n_components = n_components
        kpca_N = KernelPCA(kernel = kernel,
                               fit_inverse_transform=fit_inverse_transform,
                               gamma = gamma,
                               n_components = n_components
                               )
        
        kpca_N.fit(TrainDataScaler)    #重新训练模型
        val_n_components = kpca_N.lambdas_
        lamda = np.diag(val_n_components)            #特征值对角矩阵
        
        
        self.Kpca_Total = kpca_model
        self.Kpca_N = kpca_N
        self.samplesNum = samplesNum
        self.featuresNum = featuresNum
        self.lamda = lamda
        self.traindata = train_data
        self.K = kernel_matrix

        return self
        
    def transform(self,X,Flag = 'N'):
        '''
        获得降维后数据
        X     需要降维的数据
        Flag  若为N 则按指定特征数映射   若为Total  则按全部特征映射
        '''
        normalize = self.normalize
        dataScale = X
        
        if Flag == 'N':
            kpca_model = self.Kpca_N
        elif Flag == 'Total':
            kpca_model = self.Kpca_Total
        
        if normalize is True:
            scaler = self.Scaler
            dataScale = scaler.transform(X)
            
        data_transform = kpca_model.transform(dataScale)
        
        return data_transform 
    
    
    @staticmethod
    def Predict(static,limit):
        '''
        输入统计量和控制限
        输入预测标签
        '''
        predict = np.zeros((len(static),1))
        for i in range(len(static)):
            if static[i] >= limit:
                predict[i] = 1
            else:
                predict[i] = 0
        return predict
    @staticmethod
    
    
    def Drow_statics(T2,SPE,Ucl):
        '''
        画统计量图
        Ucl = [T2CL,SPECL]
        '''
        
        N = len(T2)

        t = np.arange(0, N)           #T2图
        plt.figure(1)
        plt.plot(t, T2)
        plt.ylabel('T2')
        plt.xlabel('Time (hr)')
        plt.title('falut diagnose based on T2')
        lineA = plt.plot(t, np.ones((N, 1))*Ucl[0], 'r--')

        plt.figure(2)                #SPE图
        plt.plot(t, SPE)
        plt.ylabel('Q')
        plt.xlabel('Time (hr)')
        plt.title('falut diagnose based on SPE')
        lineB = plt.plot(t, np.ones((N, 1))*Ucl[1], 'r--')
        plt.show()
        
        
        #%%  找标签的数目
    def get_clf_num(label):
        found_label = []
        for i in range(len(label)):
            if label[i] in found_label:
                continue
            else:
                found_label.append(label[i])
        temp = len(found_label)
        return temp
#%%  获得一组指定数目的颜色
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
 
        return hls_colors

    def RGBToHTMLColor(rgb_tuple):
        """ convert an (R, G, B) tuple to #RRGGBB """
        hexcolor = '#%02x%02x%02x' % rgb_tuple
        return hexcolor

    def ncolors(num):
        rgb_colors = []
        if num < 1:
            return rgb_colors
        hls_colors = MY_KPCA.get_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
            rgb_tuple = (r,g,b)
            temp_color = MY_KPCA.RGBToHTMLColor(rgb_tuple)
            rgb_colors.append(temp_color)
 
        return rgb_colors
    
    
    @staticmethod
    def Drow_dimension(data,label):
        '''
        画KPCA降维效果图
        '''
        num = MY_KPCA.get_clf_num(label)
        colors = MY_KPCA.ncolors(num)
        plt.figure(3)             #PCA降维数据图
        plt.xlim(data[:,0].min(),data[:,0].max())
        plt.ylim(data[:,1].min(),data[:,1].max())
        for i in range(len(data)):                                #画训练集图   由于分割数据集的方法保证了比例相同，画出来的图形会非常接近
            plt.text(data[i,0],data[i,1],str(int(label[i])),
                     color = colors[int(label[i])],
                     fontdict={'weight':'bold','size':5})
        plt.title(label='KPCA')
        plt.xlabel('feature1',fontsize=14)#x轴名称
        plt.ylabel('feature2',fontsize=14)#y轴名称
        plt.tick_params(labelsize=13) # 坐标轴字体大小
        plt.show()

        fig = plt.figure(4)       #PCA三维数据图
        ax = Axes3D(fig)
        for i in range(len(colors)):
            px = data[:,0][label == i]
            py = data[:,1][label == i]
            pz = data[:,2][label == i]
            ax.scatter(px,py,pz,c=colors[i],s=4)
        ax.set_xlabel('feature1',fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('feature2',fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('feature3',fontdict={'size': 15, 'color': 'red'})
        plt.show()
        
        
def make_moon():
    from sklearn.datasets import make_moons
    font_SimSun = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=34) # 宋体
    mpl.rcParams['font.size'] = 34 # 默认字体大小，调整10.5
    mpl.rcParams['font.serif'].insert(0,'Times New Roman') # 默认字体加上 Times New Roman
    mpl.rcParams['axes.unicode_minus'] = False
    
    x2, y2 = make_moons(n_samples=400, random_state=123)

    kpca_ = MY_KPCA(gamma=30,n_components = None,algorithm = 'kde',threshold = 0.85)
    kpca_.fit(x2)
    New_ = kpca_.transform(x2)
    plt.figure(4)
    plt.scatter(New_[y2==0, 0], New_[y2==0, 1], color='red', marker='^', alpha=0.5)
    plt.scatter(New_[y2==1, 0], New_[y2==1, 1], color='blue', marker='o', alpha=0.5)
    plt.xlabel('特征1 ',fontsize = 34,fontproperties=font_SimSun)
    plt.ylabel('特征2 ',fontsize = 34,fontproperties=font_SimSun)
    plt.tick_params(labelsize=28)
    plt.legend(['数据类型1', '数据类型2'],loc="best", prop=font_SimSun,markerscale=3)
    plt.show()
    plt.savefig('C:/Users/81479/Desktop/毕业论文/KPCA处理月形数据.svg', dpi=330)
    
    
#%%
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



    '''
    '''
    kpca = MY_KPCA(gamma=0.001,n_components = None,algorithm = 'kde',threshold = 0.6)
    TrainStart=time.time()
    kpca.fit(trainData)
    TrainEnd = time.time()
    TrainTime = TrainEnd-TrainStart
    Traindatatransform = kpca.transform(trainData)
    Testdatatransform = kpca.transform(testData)
    K = kpca.K      #这个矩阵不是正则化矩阵
    val = kpca.val
    vec = kpca.vec     #向量已经单位化
#    vec_normalize = kpca.Normalize_vec(vec)
    K_normalize = kpca.Normalize_K(K)
    n_components = kpca.n_components    #训练
    
    
    T2CL,SPECL = kpca.Get_Ucls(model = '1')
    TestStart = time.time()
    T2,SPE = kpca.Get_statics(testData,model = '1')     #统计量
        
    T2_predict = kpca.Predict(T2,T2CL)
    SPE_predict = kpca.Predict(SPE,SPECL)           #预测值
    TestEnd = time.time()
    TestTime = (TestEnd-TestStart)/testData.shape[0]
    # kpca.Drow_dimension(Testdatatransform,testLabelFC)
    ucls = [T2CL,SPECL]
    kpca.Drow_statics(T2,SPE,ucls)
    TEscore_analyse_(T2_predict,testLabelFD,testLabelFC)
    

    # make_moon()
    
    
    # time = 0
    # best_para_80 = [] 
    # best_para_70 = [] 
    # best_para_60 = [] 
    # best_para_50 = [] 
    # best_para_40 = [] 
    # best_para_30 = []
    # best_score = 0
    
    # for gamma_ in [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]:
    #     kpca = MY_KPCA(gamma=0.005,n_components = None,algorithm = 'kde',threshold = 0.85)      
    #     kpca.fit(trainData)

    #     T2CL,SPECL = kpca.Get_Ucls(model = '1')
    #     T2,SPE = kpca.Get_statics(testData,model = '1')     #统计量
        
    #     T2_predict = kpca.Predict(T2,T2CL)
    #     SPE_predict = kpca.Predict(SPE,SPECL)           #预测值
    
    #     ucls = [T2CL,SPECL]
    #     normal,fault,average = TEscore_analyse_(T2_predict,testLabelFD,testLabelFC)

    #     if normal>0.6 and fault>0.4:
    #         best_para_80.append(gamma_)
    #     elif normal>0.6 and fault>0.3:
    #         best_para_70.append(gamma_)
    #     elif normal>0.4 and fault >0.5: 
    #         best_para_60.append(gamma_)
    #     elif normal>0.4 and fault >0.4:
    #         best_para_50.append(gamma_)
    #     elif normal>0.3 and fault >0.5:
    #         best_para_40.append(gamma_)
    #     elif normal>0.3 and fault >0.4:
    #         best_para_30.append(gamma_)

            
    #     if average>best_score:
    #         BestGamma = gamma_
    #         best_score = average
    #     print('迭代次数：{}'.format(time))
    #     time = time +1

'''
Q统计量结果
*********************************************************************************************************
执行时间： Thu Apr 16 12:45:41 2020
*********************************************************************************************************
真实标签正常样本数量：9472,故障样本数量：22998
分类为正常数量：4416，SPE分类为故障数量：28054
各分类标签所含样本个数：[9472, 1950, 1640, 50, 5400, 3720, 1580, 5150, 1799, 1709]
各分类标签预测正确个数：[2368, 20950]
*********************************************************************************************************
对正常样本预测其中正确数目为：2368，对故障样本预测其中正确数目为20950
*********************************************************************************************************
分类正确总数：23318，分类错误总数：9152
*********************************************************************************************************
正常样本判断准确率：0.5362318840579711，故障样本分类准确率：0.7467740785627718
*********************************************************************************************************
平均召回率：0.7181398213735756
*********************************************************************************************************
对正常样本召回率：0.25，对故障样本召回率：0.9109487781546222
*********************************************************************************************************
对各类故障的召回率：[0.25       1.         0.98170732 0.92       0.81574074 1.
 1.         0.80213592 1.         1.        ]
*********************************************************************************************************
各类的误报率：0.75


T2统计量结果
*********************************************************************************************************
执行时间： Thu Apr 16 12:46:42 2020
*********************************************************************************************************
真实标签正常样本数量：9472,故障样本数量：22998
分类为正常数量：13797，SPE分类为故障数量：18673
各分类标签所含样本个数：[9472, 1950, 1640, 50, 5400, 3720, 1580, 5150, 1799, 1709]
各分类标签预测正确个数：[2823, 12024]
*********************************************************************************************************
对正常样本预测其中正确数目为：2823，对故障样本预测其中正确数目为12024
*********************************************************************************************************
分类正确总数：14847，分类错误总数：17623
*********************************************************************************************************
正常样本判断准确率：0.20460969776038268，故障样本分类准确率：0.6439243827986934
*********************************************************************************************************
平均召回率：0.4572528487834924
*********************************************************************************************************
对正常样本召回率：0.29803631756756754，对故障样本召回率：0.5228280720062615
*********************************************************************************************************
对各类故障的召回率：[0.29803632 0.00512821 0.81158537 0.08       0.72888889 0.37069892
 0.02151899 1.         0.10005559 0.        ]
*********************************************************************************************************
各类的误报率：0.7019636824324325
'''
