# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:28:51 2019

@author: 81479
"""

import numpy as np 
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

class Statistic:
    '''
    计算T2和SPE统计量
    '''
    @staticmethod
    def T2(data,HT2):
        '''
        输入数据和HT2矩阵
        '''
        
        data = np.square(np.linalg.norm(np.dot(HT2,data.T),axis=0))
        return data
    
    @staticmethod
    def SPE(data,HSPE):
        '''
        输入数据和SPE矩阵
        '''
        dataStat = np.zeros((data.shape[0],))
        for i in range(len(data)):
            x = data[i,:]
            dataStat[i] = np.dot(x.T,HSPE).dot(x)
        return dataStat      

    @staticmethod
    def calStatistic(data,name,**kwargs):
        """
        统计量的计算函数。
        根据不同的统计量名称调用不同的统计量求解函数，例如 T2() 或 SPE()。

        Parameters
        ----------
        data : array-like, shape [n_samples, n_features]
            待计算名为 name 的统计量的原始数据。
            其应已被标准化。
        name : str
            统计量名称
        kwargs : dict
            存放用于计算统计量值的其他数据。
            计算 'T2' 统计量，需要 HT2 数据。
            计算 'SPE' 统计量，需要 HSPE 数据。

        Returns
        ----------
        dataStat : array-like, shape [n_samples, ]
            存放着不同样本对应统计量的统计值，与样本之间存在下标对应关系。
        """
        # data应标准化之后
        dataStat = None
        if name == 'T2':
            HT2 = kwargs['HT2']
            dataStat = Statistic.T2(data,HT2)
            # dataStat = np.square(np.linalg.norm(np.dot(HT2,data.T),axis=0))
        elif name == 'SPE':
            HSPE = kwargs['HSPE']
            dataStat = Statistic.SPE(data,HSPE)
            # dataStat = np.zeros((data.shape[0],))
            # for i in range(len(data)):
            #     x = data[i,:]
            #     dataStat[i] = np.dot(x.T,HSPE).dot(x)
        else:
            raise Exception('Statistic.calStatistic(): Parameter name set wrong value:',name)
        return dataStat

    @staticmethod
    def calStatistics(data,names,tolist=False,**kwargs):
        """
        统计量的计算函数。
        根据不同的统计量名称调用不同的统计量求解函数，例如 T2() 或 SPE()。

        Parameters
        ----------
        data : array-like, shape [n_samples, n_features]
            待计算名为 name 的统计量的原始数据。
            其应已被标准化。
        names : list，默认: ['T2','SPE']
            list: 其元素必须为str，是上述可选统计量中不同值组成的列表。
        tolist: bool，默认 False
            是否将底层求取的控制限值作为列表返回，否则以 numpy.ndarray 对象返回
        kwargs : dict
            存放用于计算统计量值的其他数据。
            例如 计算 'T2' 统计量，需要 HT2 数据。

        Returns
        ----------
        dataStats : list,
            与 names 对应下标相对应的统计量计算结果，
            若 tolist == True: 其元素值为不同统计量的统计量值 float 的列表
            否则，其元素值为 numpy.ndarray 对象，其存储着不同统计量的统计量值
        """
        dataStats = []
        for name in names:
            dataStat = Statistic.calStatistic(data,name,**kwargs)
            if tolist:
                dataStat = dataStat.tolist()
            dataStats.append(dataStat)
        return dataStats
    

class UCL:
    '''
    计算控制限
    algorithm == 'formula': 公式法，可选统计量: 'T2','SPE'
    algorithm == 'kde': KDE法，只需要数据即可。
    '''
    @staticmethod
    def T2(n,k,alpha = 0.01):
        '''
        公式法T2控制限
        n:训练样本个数
        k:保留特征数
        alpha:置信度
        '''
        Ta = k * (n*n -1)/(n*(n-k)) * stats.f.isf(alpha,k,n-k)
        return Ta
    
    @staticmethod
    def SPE(eigValSorted,k,alpha=0.01):
        '''
        公式法SPE控制限
        eigValSorted：排序的特征值，包含所有特征值
        k:保留特征数
        alpha：置信度
        '''
        ca = stats.norm.isf(alpha)
        # k之后的特征值
        theta1 = np.power(eigValSorted[k:],1.0).sum()
        theta2 = np.power(eigValSorted[k:],2.0).sum()
        theta3 = np.power(eigValSorted[k:],3.0).sum()
        h0 = 1- 2*theta1*theta3/(3*theta2*theta2)
        Qa = theta1*np.power(ca*h0*np.sqrt(2*theta2)/theta1 + 1 + theta2*h0*(h0-1)/(theta1*theta1) ,1.0/h0)
        return Qa
    
    @staticmethod
    def searchKde(data,start=-4,stop=4,step=0.5,auto=False):
        """
        寻找拟合数据分布的最优窗宽，返回该最优窗宽生成的KDE。
        使用 KDE 法之前，需要通过此函数寻找最窗宽，以便得到最佳估计。

        Parameters
        ----------
        data: array-like, shape [n_samples, ]
            一维数据，待拟合的数据
        start: float
            参数搜索起始值
        stop: float
            参数搜索结束值
        step: float
            参数搜索步长。
            最终设置的参数取值方法为: np.exp2(np.arange(start,stop,step))
        auto: bool，默认 False
            True: 自动设置start、stop、step等参数，并忽略这些参数的形参值
            False: 使用形参值计算

        Returns
        -------
        kde :
            该最优窗宽生成的KDE。
        """
        data = data.reshape(-1,1)
        if auto:
            diff = (data.max() - data.min())/10.0
            start = diff/10.0
            stop = diff
            step = (stop - start) / 10
            stop += 1e-5
            print()
            print('UCL.searchKde auto')
            print('start:',start)
            print('stop:',stop)
            print('step:',step)
            bandRange = np.arange(start,stop,step)
            logEnd = np.log2(start)
            logStart = logEnd - 4
            expandRange = np.exp2(np.linspace(logStart,logEnd,4,False))
            bandRange = np.r_[expandRange,bandRange]
        else:
            bandRange = np.exp2(np.arange(start,stop,step))
            
        if start < 1e-6:
            raise Exception("The data are too small. Please don't use KDE.")
            
        params = {'bandwidth': bandRange}
        # 注意，这里使用 iid=True 去除了警告:  DeprecationWarning: The default of the `iid` parameter 
        #    will change from True to False in version 0.22 and will be removed in 0.24. 
        #    This will change numeric results when test-set sizes are unequal. DeprecationWarning
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=True)
        grid.fit(data)
        kde =  grid.best_estimator_
        bestBandWidth = kde.bandwidth
        print('BandWidth =',bestBandWidth)
        if auto:
            if bestBandWidth in bandRange[0:4]:
                print('Warning: bandwidth fall outside. Maybe search KDE by hand')
            elif bestBandWidth == bandRange[4]:
                print('Warning: bandwidth equal start. Maybe search KDE by hand')
        if bestBandWidth == bandRange.min():
            print('Warning: bandwidth equal start.')
        elif bestBandWidth == bandRange.max():
            print('Warning: bandwidth equal stop.')
        return kde

    @staticmethod
    def KDE(kde,alpha=0.01,start=-5,stop=5,step=1/10000,auto=False,data=None):
        """
        使用KDE采用小区间面积积分法估计控制限 ucl。
        KDE法。

        Parameters
        ----------
        alpha: float，默认 0.01
            控制控制限所占位置。
            上侧分位数值，一般取 0.01。
        start: float
            积分起始位置（这两个参数应设置的比数据存在区间大一些）。
        stop: float
            积分结束位置
        step: float
            小积分区间宽度，即积分面积的精确度，越小，积分值越准，但花费时间越长。
            最终设置的所有积分位置为: np.arange(start,stop,step)
        auto: bool，默认 False
            True: 自动设置start、stop、step等参数，并忽略这些参数的形参值。
            Falst: 使用形参值计算
        data: array-like, shape [n_samples, ]，默认 None
            当 auto == True 时，需传递此参数。
            data 是 searchKde() 估计时使用的数据。

        Returns
        -------
        kdeUcl : float
            KDE法计算的控制限。
        """
        if auto:
            diff = (data.max() - data.min())/2
            start = data.min() - diff
            stop = data.max() + diff
            step = (stop-start)/20000
            print()
            print('UCL.KDE auto')
            print('data min:',data.min())
            print('data max:',data.max())
            print('start:',start)
            print('stop:',stop)
            print('step:',step)
        points = np.arange(start,stop,step).reshape(-1,1)
        pdfValues = np.exp(kde.score_samples(points))
        totalArea = pdfValues.sum()
        uclArea = totalArea * (1-alpha)
        q = 0
        for i in range(len(pdfValues)):
            q += pdfValues[i]
            if (q > uclArea):
                break
        kdeUcl = start + step*i
        print("KdeUcl:",kdeUcl)
        return kdeUcl
    
    @staticmethod
    def calKdeUcl(data,alpha=0.01):
        """
        采用KDE法，计算统计量的置信限。
        寻找拟合数据分布的最优窗宽，使用该最优窗宽得到KDE，采用小区间面积积分法估计控制限 ucl。
        此函数即 algorithm == 'kde' 时的算法
        注意: 当 data 数据量低于500时，效果可能偏差很多

        Parameters
        ----------
        data: array-like, shape [n_samples, ]
            一维数据，待拟合的数据，进而根据拟合结果计算置信限值
        alpha: float，默认 0.01
            控制控制限所占位置。
            上侧分位数值，一般取 0.01。

        Returns:
        ----------
        kdeUcl: float
            KDE法计算的控制限。
        """
        kde = UCL.searchKde(data,auto=True)
        kdeUcl = UCL.KDE(kde=kde,alpha=alpha,auto=True,data=data)
        return kdeUcl
    
    @staticmethod
    def calUcl(name,alpha=0.01,**kwargs):
        """
        计算统计量的控制限。
        此函数即 algorithm == 'formula' 时的算法
        可选统计量: 'T2','SPE'

        Parameters
        ----------
        name : str
            统计量名称
        alpha: float，默认 0.01
            置信水平，常取 0.01，如需求取置信度为 0.99 的控制限，则此处填写 0.01
        kwargs : 存放用于计算控制限值的其他数据。
            统计量 'T2' 的额外参数: n,k。详见 UCL.T2()。
            统计量 'SPE' 的额外参数: val,k。详见 UCL.SPE()。

        Returns:
        ----------
        ucl: float
            formula 法计算的控制限。
        """
        ucl = None
        if name == 'T2':
            n = kwargs['n']
            k = kwargs['k']
            ucl = UCL.T2(n,k,alpha)
        elif name == 'SPE':
            val = kwargs['val']
            k = kwargs['k']
            ucl = UCL.SPE(val,k,alpha)
        else:
            raise Exception('calUcl(): Parameter name set wrong value.')
        return ucl
    
    @staticmethod
    def calUcls(names=['T2','SPE'],alpha=0.01,algorithm='formula',**kwargs):
        """
        计算多个统计量的控制限
        可选统计量: 'T2','SPE'

        Parameters
        ----------
        names : list，默认: ['T2','SPE']
            带求取控制限的统计量的名称列表
            list: 其元素必须为str，是上述可选统计量中不同值组成的列表
        alpha: float，默认 0.01
            置信水平，常取 0.01，如需求取置信度为 0.99 的控制限，则此处填写 0.01
        algorithm :  string，默认 'formula'
            采用不同的算法进行控制限求取。可填参数及含义: 
                'formula' : 采用该统计量的计算公式求解（如果存在）
                'kde' : 采用 KDE 算法求解，需额外传递 data 参数，用于求取相应数据的统计量
        kwargs : 存放用于计算控制限值的其他数据。
            当 algorithm == 'formula' 时，
                统计量 'T2' 的额外参数: n,k。详见 UCL.T2()。
                统计量 'SPE' 的额外参数: val,k。详见 UCL.SPE()。
            当 algorithm == 'kde' 时，
                统计量 'T2' 的额外参数: HT2。详见 Statistic.T2()。
                统计量 'SPE' 的额外参数: HSPE。详见 Statistic.SPE()。

        Returns:
        ----------
        ucls: list，
            列表值类型为 float
            与 names 相对应的统计量的控制限的计算结果
        """
        ucls = []
        if algorithm == 'kde':
            data = kwargs['data']
            del kwargs['data']
            dataStats = Statistic.calStatistics(data,names=names,**kwargs)
            for dataStat in dataStats:
                kdeUcl = UCL.calKdeUcl(dataStat,alpha)
                ucls.append(kdeUcl)
        elif algorithm == 'formula':
            for name in names:
                ucl = UCL.calUcl(name,alpha,**kwargs)
                ucls.append(ucl)
        else:
            raise Exception('UCL.calUcls(): Parameter algorithm set wrong value.')
        return ucls

    @staticmethod
    def judge(valueList,ucl,tolist=False):
        """
        根据 ucl 预测 valueList 中的值是否是故障，返回判断结果的标签，1是故障，0不是故障。
        将 valueList 向量中，值大于ucl的位置标志为1。

        Parameters
        ----------
        valueList : array-like, shape [n_samples,]
            不同样本的统计量值。
        ucl: float
            用于判断是否是故障的控制限。
        tolist: bool，默认 False
            返回 列表 或是 numpy.ndarray 对象。

        Returns:
        ----------
        predictLabels : float
            若 tolist == True: 其元素值为不同样本的是否有故障的判定结果的列表
            否则，其元素值为 numpy.ndarray 对象，其存储着不同样本的是否有故障的判定结果
        """
        valueList = np.array(valueList)
        predictResult = (valueList >= ucl)
        predictLabels = predictResult.astype(float)
        if tolist:
            predictLabels = predictLabels.tolist()
        # length = len(valueList)
        # predictLabels = np.zeros(shape=(length,),dtype=float)
        # for i in range(length):
        #     if valueList[i] >= ucl:
        #         predictLabels[i] = 1.0
        return predictLabels       