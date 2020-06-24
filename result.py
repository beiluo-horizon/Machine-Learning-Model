# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:15:37 2019

@author: 81479
"""

import colorsys
import random
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import time
import pandas as pd

#%%求标签数目
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
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_tuple = (r,g,b)
        temp_color = RGBToHTMLColor(rgb_tuple)
        rgb_colors.append(temp_color)
 
    return rgb_colors
#%%结果分析
def score_analyse(predictlabel,label_use,model = 'Hold-out',CV_parameter = None):
    
    if model == 'Hold-out':
        num = get_clf_num(label_use)
    elif model == 'CV':
        num = get_clf_num(label_use[0])
    each_true_uselabel = [0]*num     #计数真实标签各分类各有多少
    each_predictlabel = [0]*num
    predictright_NUM = 0
    predicterror_NUM = 0
    predictright_Index = []
    predicterror_Index = []
    each_predictright = [0]*num
    each_predicterror = [0]*num
    eachPrecision = [0]*num
    eachclfRecall = [0]*num
    averageRecall = 0
    averagePrecision = 0
    each_information = np.zeros((num,num))
    if model =='Hold-out':
    
        each_true_uselabel = [0]*num     #计数真实标签各分类各有多少
        each_predictlabel = [0]*num
        predictright_NUM = 0
        predicterror_NUM = 0
        predictright_Index = []
        predicterror_Index = []
        each_predictright = [0]*num
        each_predicterror = [0]*num
        eachPrecision = [0]*num
        eachclfRecall = [0]*num
    
        for i in range(len(label_use)):
            currentpredictlabel = int(predictlabel[i])  #获得当前的标签预测值
            currentlabel_use = int(label_use[i])        #获得真实的多分类标签值
            each_true_uselabel[currentlabel_use]+=1     #真实对应类计数
            each_predictlabel[currentpredictlabel]+=1    #预测各类计数
            #对结果的评估
            if currentpredictlabel == currentlabel_use:   #分类正确
                predictright_NUM +=1                       #预测正确样本数目
                predictright_Index.append(i)                #获取预测正确的样本下标
                each_predictright[currentpredictlabel]+=1      #各标签预测正确数目
            else:
                predicterror_NUM+=1                      #分类错误总数
                predicterror_Index.append(i)            #分类错误下标
                each_predicterror[currentlabel_use]+=1      #各标签错误分类数目
            
    #求召回率
        eachclfRecall = np.divide(each_predictright,each_true_uselabel)        #针对多分类的召回率
        averageRecall = np.divide(predictright_NUM,predictlabel.shape[0])    #平均召回率
        eachPrecision = np.divide(each_predictright,each_predictlabel)          #针对多分类的精确度
    
        total= 0.0
        for t in range(len(eachPrecision)):
            total += eachPrecision[t]
        averagePrecision = np.divide(total,num)
    
        each_information = np.zeros((num,num))
        for i in range(len(label_use)):
            currentpredictlabel = int(predictlabel[i])  #获得当前的标签预测值
            currentlabel_use = int(label_use[i])        #获得真实的多分类标签值
            each_information[currentpredictlabel,currentlabel_use] = 1 + each_information[currentpredictlabel,currentlabel_use]
        
        averageScore = predictright_NUM/(predictright_NUM+predicterror_NUM)
    if model =='CV':
        each_true_uselabel_grid = np.zeros((CV_parameter,num))
        each_predictlabel_grid = np.zeros((CV_parameter,num))
        predictright_NUM_grid = [0]*CV_parameter
        predicterror_NUM_grid = [0]*CV_parameter
        each_predictright_grid = np.zeros((CV_parameter,num))
        each_predicterror_grid = np.zeros((CV_parameter,num))
        eachPrecision_grid = np.zeros((CV_parameter,num))
        eachclfRecall_grid = np.zeros((CV_parameter,num))
        for m in range(int(CV_parameter)):
            num = get_clf_num(label_use)
            each_true_uselabel = [0]*num     #计数真实标签各分类各有多少
            each_predictlabel = [0]*num
            predictright_NUM = 0
            predicterror_NUM = 0
            predictright_Index = []
            predicterror_Index = []
            each_predictright = [0]*num
            each_predicterror = [0]*num
            eachPrecision = [0]*num
            eachclfRecall = [0]*num
    
            for i in range(len(label_use)):
                currentpredictlabel = int(predictlabel[i])  #获得当前的标签预测值
                currentlabel_use = int(label_use[i])        #获得真实的多分类标签值
                each_true_uselabel[currentlabel_use]+=1     #真实对应类计数
                each_predictlabel[currentpredictlabel]+=1    #预测各类计数
                #对结果的评估
                if currentpredictlabel == currentlabel_use:   #分类正确
                    predictright_NUM +=1                       #预测正确样本数目
                    predictright_Index.append(i)                #获取预测正确的样本下标
                    each_predictright[currentpredictlabel]+=1      #各标签预测正确数目
                else:
                    predicterror_NUM+=1                      #分类错误总数
                    predicterror_Index.append(i)            #分类错误下标
                    each_predicterror[currentlabel_use]+=1      #各标签错误分类数目
            
            #求召回率
            eachclfRecall = np.divide(each_predictright,each_true_uselabel)        #针对多分类的召回率
            averageRecall = np.divide(predictright_NUM,predictlabel.shape[0])    #平均召回率
            eachPrecision = np.divide(each_predictright,each_predictlabel)          #针对多分类的精确度
    
            total= 0.0
            for t in range(len(eachPrecision)):
                total += eachPrecision[t]
            averagePrecision = np.divide(total,num)
    
            each_information = np.zeros((num,num))
            for i in range(len(label_use)):
                currentpredictlabel = int(predictlabel[i])  #获得当前的标签预测值
                currentlabel_use = int(label_use[i])        #获得真实的多分类标签值
                each_information[currentpredictlabel,currentlabel_use] = 1 + each_information[currentpredictlabel,currentlabel_use]
                
            each_true_uselabel_grid[m,:] = each_true_uselabel
            each_predictlabel_grid[m,:] = each_predictlabel
            predictright_NUM_grid[m] = predictright_NUM
            predicterror_NUM_grid[m] = predicterror_NUM
            each_predictright_grid[m,:] = each_predictright
            each_predicterror_grid[m,:] = each_predicterror
            eachPrecision_grid[m,:] = eachPrecision
            eachclfRecall_grid[m,:] = eachclfRecall
            
        predictright_NUM = np.mean(predictright_NUM_grid)
        predicterror_NUM = np.mean(predicterror_NUM_grid)
        each_true_uselabel = np.mean(each_true_uselabel_grid,axis = 0)
        each_predictlabel = np.mean(each_predictlabel_grid,axis = 0)
        each_predictright = np.mean(each_predictright_grid,axis = 0)
        each_predicterror = np.mean(each_predicterror_grid,axis = 0)
        eachPrecision = np.mean(eachPrecision_grid,axis = 0)
        eachclfRecall = np.mean(eachclfRecall_grid,axis = 0)
            
            
    score_dict = {
            'each_true_uselabel':each_true_uselabel,
            'each_predictlabel':each_predictlabel,
            'each_predictright':each_predictright,
            'each_predicterror':each_predicterror,       #正确和错误相加=各分类标签所含样本个数    #饼状
            'predictright_NUM':predictright_NUM,
            'predicterror_NUM':predicterror_NUM,                  #分类正确总数+分类错误总数 = 样本总数    #饼状图
            'averagePrecision':averagePrecision,
            'eachPrecision':eachPrecision,            #柱状图
            'averageRecall':averageRecall,
            'eachclfRecall':eachclfRecall,                 #柱状图
            'each_information':each_information,
            'averageScore':averageScore,
            }
    
    
    
    return score_dict
    


def  Drow_pic (label_use,score_dict,Drow = False):
    
    num = get_clf_num(label_use)
    '''
    画柱状图
    '''
    if Drow is True:
        colors = ncolors(2)
        name_list=['Cla_0']
        for t in range(num):
            if t == 0:
                continue
            else:
                name_now = 'Cla_'+str(t)
                name_list.append(name_now)
        Precision_list = score_dict['eachPrecision']
        Recall_list= score_dict['eachclfRecall']
        plt.figure(figsize=(10,10))
        x = list(range(len(Precision_list)))
        total_width, n = 0.8, 2
        width = total_width / n
        plt.bar(x, Precision_list, width=width, label='Accuracy', tick_label=name_list,fc=colors[0])
        for i in range(len(x)):
            x[i] += width
        plt.bar(x, Recall_list, width=width, label='Recall',tick_label=name_list,fc=colors[1])
        plt.legend(labels=['Accuracy', 'Recall'],loc='upper left',fontsize=13)   
        plt.show()
        #饼状图
        each_true_uselabel = score_dict['each_true_uselabel']
        each_predictright = score_dict['each_predictright']
        each_predicterror = score_dict['each_predicterror']
        for i in range(len(each_true_uselabel)):
            # 保证圆形
            plt.figure(figsize=(10,10))
            name_now = 'category_'+str(i)
            plt.title(label=name_now)
            list_pie = [each_predictright[i],each_predicterror[i]]
            name_pie = ['predictright','predicterror']
            plt.axes(aspect=1)
            plt.pie(x=list_pie, labels=name_pie, autopct='%3.1f %%')
            plt.show()
        #总样本饼状图
        predictright_NUM = score_dict['predictright_NUM']
        predicterror_NUM = score_dict['predicterror_NUM']
        plt.figure(figsize=(10,10))
        plt.title(label='total_sample')
        list_pie = [predictright_NUM,predicterror_NUM]
        name_pie = ['predictright_NUM','predicterror_NUM']
        plt.axes(aspect=1)
        plt.pie(x=list_pie, labels=name_pie, autopct='%3.1f %%')
        plt.show()
    
    #信息表格
    titles_COL = ['']*num
    titles_ROW = ['']*num
    for m in range (len(titles_COL)):
        title = 'pre_'+str(m)
        titles_COL[m] = title
        title = 'true_'+str(m)
        titles_ROW[m] = title
    data_frame = score_dict['each_information']
    data_frame = pd.DataFrame(data_frame,titles_COL,titles_ROW)
    
    pd.set_option('display.max_rows',1000)   # 具体的行数或列数可自行设置
    pd.set_option('display.max_columns',1000)
    pd.set_option('display.width',200)
    
    
    #一些文字的说明
    print("*********************************************************************************************************")
    print('执行时间：',time.asctime())
    print("*********************************************************************************************************")      
    #打印结果
    print('各分类标签所含样本个数：{}'.format(score_dict['each_true_uselabel']))
    print('预测为各分类标签数目：{}'.format(score_dict['each_predictlabel']))
    print('各分类标签预测正确数目：{}'.format(score_dict['each_predictright']))
    print('各分类标签预测错误数目：{}'.format(score_dict['each_predicterror']))
    print("*********************************************************************************************************")
    print('分类正确总数：{}，分类错误总数：{}'.format(score_dict['predictright_NUM'],score_dict['predicterror_NUM']))
    print('分类成绩：{}'.format(score_dict['averageScore']))
    print("*********************************************************************************************************")
    print('准确率公式：判断为该类正确格式/判断为该类总数')
    print('平均准确率：{}'.format(score_dict['averagePrecision']))
    print('对各类故障的判断准确率：{}'.format(score_dict['eachPrecision']))
    print("*********************************************************************************************************")
    print('召回率公式：该类分类正确总数/该类样本数')
    print('平均召回率：{}'.format(score_dict['averageRecall']))               
    print('各类故障的召回率：{}'.format(score_dict['eachclfRecall']))
    print("*********************************************************************************************************")
    print(data_frame)
    
def score_analyse_(Predictlabel,LabelFC,model = 'Hold-out',CV_parameter = None):
    if model == 'Hold-out':
        num = get_clf_num(LabelFC)
    elif model == 'CV':
        num = get_clf_num(LabelFC[0])
        
    EachTrueLabelFC = [0]*num     #计数真实标签各分类各有多少
    EachPredictLable = [0]*num    #计数预测分类各有多少
    CurrentPredictLabel = 0       #当前预测值

    PredictRightNum = 0         #分类正确总数
    PredictErrorNum = 0         #分类错误总数
    PredictRightIndex = []         #分类正确的下标
    PredictErrorIndex = []         #分类错误的下标
    EachPredictRight = [0]*num         #各分类预测正确数目
    EachPredictError = [0]*num       #各分类预测错误数目
    
    
    FP = 0
    TN = 0
    
    if model =='Hold-out':   
        EachPrecision = [0]*num
        EachRecall = [0]*num
        EachF1 = [0]*num
    
        for i in range(len(LabelFC)):
            CurrentPredictLabel = int(Predictlabel[i])  #获得当前的标签预测值
            CurrentLabel = int(LabelFC[i])             #获得真实的多分类标签值
            EachTrueLabelFC[CurrentLabel]+=1     #真实对应类计数
            EachPredictLable[CurrentPredictLabel]+=1    #预测各类计数
            #对结果的评估
            if CurrentPredictLabel == CurrentLabel:   #分类正确
                PredictRightNum +=1                       #预测正确样本数目
                PredictRightIndex.append(i)                #获取预测正确的样本下标
                EachPredictRight[CurrentPredictLabel]+=1      #各标签预测正确数目                
                if CurrentLabel == 0:    #实际情况为0
                    TN+=1
            else:
                PredictErrorNum+=1                      #分类错误总数
                PredictErrorIndex.append(i)            #分类错误下标
                EachPredictError[CurrentPredictLabel]+=1      #各标签错误分类数目
                
                if CurrentLabel == 0:    #实际情况为0
                    FP+=1 
            
    #求召回率
        EachRecall = np.divide(EachPredictRight,EachTrueLabelFC)        #针对多分类的召回率
        AverageRecall = np.divide(PredictRightNum,Predictlabel.shape[0])    #平均召回率
        EachPrecision = np.divide(EachPredictRight,EachPredictLable)          #针对多分类的精确度
        EachF1 = 2*EachRecall*EachPrecision/(EachRecall+EachPrecision)
       
        each_information = np.zeros((num,num))
        for i in range(len(LabelFC)):
            currentpredictlabel = int(Predictlabel[i])          #获得当前的标签预测值
            currentlabel_use = int(LabelFC[i])                #获得真实的多分类标签值
            each_information[currentpredictlabel,currentlabel_use] = 1 + each_information[currentpredictlabel,currentlabel_use]
        
        FARSUM = TN + FP
        FAR = np.divide(FP,FARSUM)
        
        score_dict = {
                'EachTrueLabelFC':EachTrueLabelFC,
                'EachPredictLable':EachPredictLable,
                'EachPredictRight':EachPredictRight,
                'EachPredictError':EachPredictError,       
                'PredictRightNum':PredictRightNum,
                'PredictErrorNum':PredictErrorNum,                  
                'EachPrecision':EachPrecision,    
                'EachF1':EachF1,
                'AverageRecall':AverageRecall,
                'EachRecall':EachRecall,                 
                'each_information':each_information,
                'FAR':FAR
                }
        
    if model =='CV':
        
        EachTrueLabelFC_CV = np.zeros((CV_parameter,num))     #计数真实标签各分类各有多少
        EachPredictLable_CV = np.zeros((CV_parameter,num))    #计数预测分类各有多少
        CurrentPredictLabel = 0       #当前预测值

    
        PredictRightNum_CV = [0]*CV_parameter         #分类正确总数
        PredictErrorNum_CV = [0]*CV_parameter         #分类错误总数
        EachPredictRight_CV = np.zeros((CV_parameter,num))         #各分类预测正确数目
        EachPredictError_CV = np.zeros((CV_parameter,num))       #各分类预测错误数目
        EachPrecision_CV = np.zeros((CV_parameter,num))
        EachRecall_CV = np.zeros((CV_parameter,num))
        AverageRecall_CV = [0]*CV_parameter
        
        FP = [0]*CV_parameter
        TN = [0]*CV_parameter
        FAR = [0]*CV_parameter
        FARSUM = [0]*CV_parameter
        
        for m in range(int(CV_parameter)):     #对每一折考虑
            PredictLabel_CV = Predictlabel[m]
            LabelFC_CV = LabelFC[m]
            for i in range(len(LabelFC_CV)):
                CurrentPredictLabel = int(PredictLabel_CV[i])  #获得当前的标签预测值
                CurrentLabel = int(LabelFC_CV[i])        #获得真实的多分类标签值
                EachTrueLabelFC_CV[m,CurrentLabel]+=1     #真实对应类计数
                EachPredictLable_CV[m,CurrentPredictLabel]+=1    #预测各类计数
                #对结果的评估
                if CurrentPredictLabel == CurrentLabel:   #分类正确
                    PredictRightNum_CV[m]+=1                       #预测正确样本数目
                    EachPredictRight_CV[m,CurrentPredictLabel]+=1      #各标签预测正确数目
                    if CurrentLabel == 0:    #实际情况为0
                        TN[m]+=1
                else:
                    PredictErrorNum_CV[m]+=1                      #分类错误总数
                    EachPredictError_CV[m,CurrentLabel]+=1      #各标签错误分类数目
                    if CurrentLabel == 0:    #实际情况为0
                        FP[m]+=1 
            
            #求召回率
            EachRecall_CV[m,:] = np.divide(EachPredictRight_CV[m,:],EachTrueLabelFC_CV[m,:])        #针对多分类的召回率
            AverageRecall_CV[m] = np.divide(PredictRightNum_CV[m],PredictLabel_CV.shape[0])    #平均召回率
            EachPrecision_CV[m,:] = np.divide(EachPredictRight_CV[m,:],EachPredictLable_CV[m,:])          #针对多分类的精确度
            
            FARSUM[m] = TN[m] + FP[m]        
            FAR[m] = np.divide(FP[m],FARSUM[m])
            
        PredictRightNum = np.mean(PredictRightNum_CV)
        PredictErrorNum = np.mean(PredictErrorNum_CV)
        EachTrueLabelFC = np.mean(EachTrueLabelFC_CV,axis = 0)
        EachPredictLable = np.mean(EachPredictLable_CV,axis = 0)
        EachPredictRight = np.mean(EachPredictRight_CV,axis = 0)
        EachPredictError = np.mean(EachPredictError_CV,axis = 0)
        EachPrecision = np.mean(EachPrecision_CV,axis = 0)
        EachRecall = np.mean(EachRecall_CV,axis = 0)
        EachF1 = 2*EachRecall*EachPrecision/(EachRecall+EachPrecision)
        AverageRecall = np.mean(AverageRecall_CV)
        FAR = np.mean(FAR)    
            
        score_dict = {
            'EachTrueLabelFC':EachTrueLabelFC,
            'EachPredictLable':EachPredictLable,
            'EachPredictRight':EachPredictRight,
            'EachPredictError':EachPredictError,       
            'PredictRightNum':PredictRightNum,
            'PredictErrorNum':PredictErrorNum,                  
            'EachPrecision':EachPrecision,            
            'AverageRecall':AverageRecall,
            'EachF1':EachF1,
            'EachRecall':EachRecall,
            'FAR':FAR              
            }
        
        
    #一些文字的说明
    print("*********************************************************************************************************")
    print('执行时间：',time.asctime())
    print("*********************************************************************************************************")      
    #打印结果
    print('各分类标签所含样本个数：{}'.format(score_dict['EachTrueLabelFC']))
    print('预测为各分类标签数目：{}'.format(score_dict['EachPredictLable']))
    print('各分类标签预测正确数目：{}'.format(score_dict['EachPredictRight']))
    print('各分类标签预测错误数目：{}'.format(score_dict['EachPredictError']))
    print("*********************************************************************************************************")
    print('分类正确总数：{}，分类错误总数：{}'.format(score_dict['PredictRightNum'],score_dict['PredictErrorNum']))
    print('分类成绩：{}'.format(score_dict['AverageRecall']))
    print("*********************************************************************************************************")
    print('准确率公式：判断为该类正确格式/判断为该类总数')
    print('对各类故障的判断准确率：{}'.format(score_dict['EachPrecision']))
    print("*********************************************************************************************************")
    print('召回率公式：该类分类正确总数/该类样本数')
    print('平均召回率：{}'.format(score_dict['AverageRecall']))               
    print('各类故障的召回率：{}'.format(score_dict['EachRecall']))
    print("*********************************************************************************************************")
    print('F1公式：2*召回率*准确率/（召回率+准确率）')
    print('F1:{}'.format(score_dict['EachF1']))
    print("*********************************************************************************************************")
    print('误判率:{}'.format(score_dict['FAR']))
    print("*********************************************************************************************************")
    if model == 'Hold-out':
        #信息表格
        titles_COL = ['']*num
        titles_ROW = ['']*num
        for m in range (len(titles_COL)):
            title = 'pre_'+str(m)
            titles_COL[m] = title
            title = 'true_'+str(m)
            titles_ROW[m] = title
        data_frame = score_dict['each_information']
        data_frame = pd.DataFrame(data_frame,titles_COL,titles_ROW)
        
        pd.set_option('display.max_rows',1000)   # 具体的行数或列数可自行设置
        pd.set_option('display.max_columns',1000)
        pd.set_option('display.width',200)
        print(data_frame)
    
    
    return score_dict

def result_process(original_test_label,predict_label,Drow = False):
    score_dict = score_analyse(predict_label,original_test_label)
    Drow_pic(original_test_label,score_dict,Drow= Drow)  #画图并保存
    
    return score_dict['averageRecall'],score_dict['eachclfRecall']
    
if __name__ == '__main__':
    pass
    
    
        
        
        
    

             

