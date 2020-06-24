# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:25:43 2019

@author: 81479
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
import colorsys
import random
from Ucl import UCL,Statistic
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import StratifiedKFold   #保证每折中样本比例相同
from pylab import mpl
from matplotlib.font_manager import FontProperties
import time

class MY_PCA:
    '''
    封装PCA
    '''
    
    def __init__(self,n_components=None,normalize=True,threshold=0.85,
                 alpha=0.01,algorithm='kde'):
        '''
        n_components:保留的特征数，None则根据threshold选择特征占比
        normalize：为True则进行标准化
        alpha：统计量的参数
        
        '''
        
        self.n_components = n_components
        self.normalize = normalize
        self.threshold = threshold
        self.alpha = alpha
        self.algorithm = algorithm
        self.ucls = None
    
    @staticmethod
    def decompose(trainDataScale):
        """
        将 trainDataScale 的协方差进行特征分解，得到特征值，特征向量，
        并对特征值按从大到小的顺序排序（对应的特征向量也排序）

        Parameters
        ----------
        trainDataScale : array-like, shape [n_samples, n_features]
            将 trainDataScale 的协方差矩阵进行特征分解。

        Returns
        -------
        val : array-like, shape [n_eigenvalues]
            从大到小排序的特征值，n_eigenvalues == n_features
        vec : array-like, shape [n_features, n_eigenvalues]
            特征向量，是列向量
            val[i] 特征值 对应的特征向量为 vec[:,i]。
        """
        covariance = np.cov(trainDataScale.T)
        # 特征值，特征向量矩阵（特征向量是列向量）
        val,vec = np.linalg.eig(covariance)
        # 注意：这里复数只取了实数部分
        val = val.real.astype(float)
        vec = vec.real.astype(float)
        order = val.argsort()
        val = np.flip(val[order],0)
        vec = np.flip(vec[:,order],1)    #按列排列
        return val,vec
    
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
    
    def split (data,P):
        '''
        分割数据集
        '''
        PPT = np.dot(P,P.T)
        X_ = np.dot(data,PPT)
        E_ = data - X_
        return X_,E_
    
    def _split(self,data):
        normalize = self.normalize
        dataScale = data
        if normalize is True:
            scaler = self.scaler
            dataScale = scaler.transform(data)
        P = self.P
        PPT = np.dot(P,P.T)
        X_ = np.dot(dataScale,PPT)
        E_ = dataScale - X_
        return X_,E_
    
    @staticmethod
    def selectComponents(vec,n_components):
        """
        从 vec 中，选取前 n_components 个特征向量，要求 vec 中特征向量排列顺序对应从大到小的特征值。

        Parameters
        ----------
        vec : array-like, shape [n_features, n_features]
            特征向量矩阵

        Returns
        -------
        P : array-like, shape [n_features, n_components]
            变换矩阵
        """
        P = vec[:,0:n_components]
        return P
    
    
    def fit(self,trainData,splitX = False,UCLFlag = True):
        '''
        PCA训练过程
        '''
        n_components = self.n_components
        normalize = self.normalize
        threshold = self.threshold
        samplesNum,featuresNum = trainData.shape
        trainDataScale = trainData
        scaler = None
        if normalize is True:
            scaler = preprocessing.StandardScaler().fit(trainData)
            trainDataScale = scaler.transform(trainData)
        val,vec = MY_PCA.decompose(trainDataScale)
        if n_components is None:
            n_components = MY_PCA.calComponentSelectNums(val,threshold=threshold)
        P = MY_PCA.selectComponents(vec,n_components)
        X_ = None
        E_ = None
        if splitX:
            X_,E_ = MY_PCA.split(trainDataScale,P)
        self.trainData = trainData
        self.trainDataScale = trainDataScale
        self.scaler = scaler
        self.samplesNum = samplesNum
        self.featuresNum = featuresNum
        self.val = val
        self.vec = vec
        self.scaler = scaler
        self.n_components = n_components
        self.P = P
        self.X_ = X_
        self.E_ = E_
        self.tempMatrix = dict()
        if UCLFlag is True:
            self.ucls = self.calUcls(alpha=self.alpha,algorithm=self.algorithm)
#        return self
        return val,vec
    
    def transform(self,X):
        '''
        获得降维后数据
        '''
        normalize = self.normalize
        P = self.P
        dataScale = X
        if normalize is True:
            scaler = self.scaler
            dataScale = scaler.transform(X)
        return np.dot(dataScale,P)    
    
    def _getTempMatrix(self):
        """
        获取用于方便计算统计量的临时计算矩阵
        目前可用统计量名称：'T2','SPE',
        如果对应统计量没有临时矩阵，或者统计量不存在，则返回 None

        Parameters
        ----------
        name : str
            统计量名称

        Returns
        -------
        matrix : None 或 array-like, shape [n_features, n_features]
            None :如果对应统计量没有临时矩阵，或者统计量不存在时
            array-like : 对应统计量的临时计算矩阵
        """
        n_components = self.n_components
        val = self.val
        P = self.P
        S = np.diag(1.0/np.sqrt(val[0:n_components]))
        HT2 = np.dot(S,P.T)
        self.tempMatrix['T2'] = HT2
        P = self.P
        PPT = np.dot(P,P.T)
        HSPE = np.identity(P.shape[0]) - PPT
        self.tempMatrix['SPE'] = HSPE
        return self.tempMatrix       
    
    def calUcls(self,alpha=0.01,algorithm='kde'):
        """
        计算多个统计量的控制限
        可选统计量：'T2','SPE'

        Parameters
        ----------
        names: list，默认：['T2','SPE']
            待求取控制限的统计量的名称列表
            list: 其元素必须为str，是上述可选统计量中不同值组成的列表
        alpha: float，默认 0.01
            置信水平，常取 0.01，如需求取置信度为 0.99 的控制限，则此处填写 0.01
        algorithm ： string，默认 'kde'
            采用不同的算法进行控制限求取。可填参数及含义：
            'formula' : 采用该统计量的计算公式求解（如果存在）
            'kde' : 采用 KDE 算法求解

        Returns:
        ----------
        ucls: list，
            列表值类型为 float
            与 names 相对应的统计量的控制限的计算结果
        """
        names=['T2','SPE']
        if algorithm == 'kde':
            dataScale = self.trainDataScale
            Temp = self._getTempMatrix()
            kw = dict()
            for name in names:
                kw['H' + name] = Temp[name]
            ucls = UCL.calUcls(names=names,alpha=alpha,algorithm='kde',
                        data=dataScale,
                        **kw)
        elif algorithm == 'formula':
            n = self.samplesNum
            k = self.n_components
            val = self.val
            ucls = UCL.calUcls(names=names,alpha=alpha,algorithm='formula',
                        n=n,
                        k=k,
                        val=val)
        else:
            raise Exception('MyPca.calUcls(): Parameter algorithm set wrong value.')
        self.ucls = ucls
        return ucls
        
        
    def calStatistics(self,data,names=['T2','SPE'],tolist=False):
        """
        计算数据 data 的在某些统计量下的统计量值
        可选统计量：'T2','SPE'

        Parameters
        ----------
        data : array-like, shape [n_samples, n_features]
            待计算名为 name 的统计量的原始数据。
            其应已被标准化。
        names: list，默认：['T2','SPE']
            带求取统计量值的统计量的名称列表
            list: 其元素必须为str，是上述可选统计量中不同值组成的列表
        tolist: bool，默认 False
            是否将底层求取的控制限值作为列表返回，否则以 numpy.ndarray 对象返回

        Returns:
        ----------
        dataStats: list
            与 names 对应下标相对应的统计量计算结果，
            若 tolist == True: 其元素值为不同统计量的统计量值 float 的列表
            否则，其元素值为 numpy.ndarray 对象，其存储着不同统计量的统计量值
        """
        # HT2 = self._getTempMatrix('T2')
        # HSPE = self._getTempMatrix('SPE')
        normalize = self.normalize
        dataScale = data
        Temp = self._getTempMatrix()
        kw = dict()
        for name in names:
            kw['H' + name] = Temp[name]
        if normalize is True:
            scaler = self.scaler
            dataScale = scaler.transform(data)
        dataStats = Statistic.calStatistics(dataScale,names=names,tolist=tolist,
                                            **kw)
        return dataStats
    
    @staticmethod
    def Predict(Statistic,Ucl):
        '''
        根据统计量获得预测结果
        '''
        predictLabel = np.zeros((Statistic.shape))
        for i in range(predictLabel.shape[0]):
            if Statistic[i] >= Ucl:
                predictLabel[i] = 1
            else:
                predictLabel[i] = 0
        return predictLabel
    @staticmethod
    def Drow_statics(T2,SPE,Ucl):
        '''
        画统计量图
        '''
        
        N = T2.shape[0]
        legend_Name = ['statisitcs','control limit']
        t = np.arange(0, N)           #T2图
        plt.figure(1)
        plt.plot(t, T2)
        plt.ylabel('T2',fontsize = 14)
        plt.xlabel('Time (hr)',fontsize = 14)
        plt.title('falut detection based on T2',fontsize = 14)
        lineA = plt.plot(t, np.ones((N, 1))*Ucl[0], 'r--')
        plt.legend(legend_Name,loc='upper right',fontsize=14) 
        plt.ylim(0,6000)
        mpl.rcParams['font.serif'].insert(0,'Times New Roman')
        plt.tight_layout()
        plt.show()

        plt.figure(2)                #SPE图
        plt.plot(t, SPE)
        plt.ylabel('Q',fontsize = 14)
        plt.xlabel('Time (hr)',fontsize = 14)
        plt.title('falut detection based on SPE',fontsize = 14)
        lineB = plt.plot(t, np.ones((N, 1))*Ucl[1], 'r--')
        plt.legend(legend_Name,loc='upper right',fontsize=14) 
        plt.ylim(0,800)
        mpl.rcParams['font.serif'].insert(0,'Times New Roman')
        plt.tight_layout()
        plt.show()
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
        hls_colors = MY_PCA.get_n_hls_colors(num)
        for hlsc in hls_colors:
            _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
            r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
            rgb_tuple = (r,g,b)
            temp_color = MY_PCA.RGBToHTMLColor(rgb_tuple)
            rgb_colors.append(temp_color)
 
        return rgb_colors
    
    @staticmethod
    def Drow_dimension(data,label):
        '''
        画PCA降维效果图
        '''
        font_SimSun = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=24) # 宋体
        mpl.rcParams['font.size'] = 24 # 默认字体大小，调整10.5
        mpl.rcParams['font.serif'].insert(0,'Times New Roman') # 默认字体加上 Times New Roman
        mpl.rcParams['axes.unicode_minus'] = False
        
        num = MY_PCA.get_clf_num(label)
        colors = MY_PCA.ncolors(num)
        plt.figure(3)             #PCA降维数据图

        plt.xlim(data[:,0].min(),data[:,0].max())
        plt.ylim(data[:,1].min(),data[:,1].max())
        # plt.xlim(data[:,0].min(),300)
        # plt.ylim(data[:,1].min(),250)
        for i in range(len(data)):                                #画训练集图   由于分割数据集的方法保证了比例相同，画出来的图形会非常接近
            plt.text(data[i,0],data[i,1],str(int(label[i])),
                     color = colors[int(label[i])],
                     fontdict={'weight':'bold','size':5})
        plt.xlabel('特征1',fontsize = 24,fontproperties=font_SimSun)#x轴名称
        plt.ylabel('特征2',fontsize = 24,fontproperties=font_SimSun)#y轴名称
        plt.tick_params(labelsize=15) # 坐标轴字体大小
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
        

     
if __name__ == '__main__':
    pass