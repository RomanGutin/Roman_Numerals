# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 20:46:00 2019

@author: Roman
"""
###Main Interface for Running Model###
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV


''' Dictionaries to save data '''
round_data = {}     
loop_dict= {}
var_dict= {}
ElasticNet_dict= {}
''' '''
''' Model Inputs'''
Run_name =  "Zero_Padding"
l1_ratios = [.1,.2,.3,4.,5,.6,.75,.85,.88,.90,.91,.92,.93,.94,.95,.96,.97,.98,.99,1] 
Encoding = Zero_Padding            # Encdoing              Found in Load_Objects.py ##
Encoding['pMeas']= nine_pep['pMeas']     ## Tag  on pIC50's ##
Scores = scores_ZeroPad      # Scores Dictionary     Found in Load_Objects.py ##
step_size = 1                         
''' '''

def Model(Encoding,Scores,Run_name,step_size,loop_dict,var_dict,round_data,ElasticNet_dict,l1_ratios,All_data):
    Pearson_correlations= []
    Data = Encoding.copy() #copy, so it does not change#
    Data_sets = CV_split(Data,5)  # The Big 5# 
    for cv_round in range(len(Data_sets)):
        score_dict= Scores.copy() #Randomized scores at the start each time#
        Test_set= Data_sets[cv_round]  
        Train_set = exclude(Data_sets,cv_round)                         #Keeps everything but the train set#
        Train_set = pd.concat(Train_set)                                #All train sets into on dataframe#
        X = Train_set.iloc[:,:Train_set.shape[1]-1]                     #features#
        X['Intercept']= 1                                               #add intercept#
        y= pd.DataFrame(Train_set['pMeas'])                             #targets#
        AM_EndOfLoopError= []
        AM_EndOfLoopError.append(Get_Error(X,y,score_dict))             # The Error Before AM Tuning #
       
        """AM Tuning Looping Starts Here and Adds a value to End of Loop Error"""
        Loop_num = 1 #
        AM_EndOfLoopError.append(Amplitude_Tuning(X,y,step_size,score_dict,Loop_num,Run_name,cv_round,loop_dict,var_dict))
        round_data[cv_round] = loop_dict
        while ((AM_EndOfLoopError[-1]-AM_EndOfLoopError[-2])/(AM_EndOfLoopError[-2])) < -0.001:
            Loop_num += 1
            AM_EndOfLoopError.append(Amplitude_Tuning(X,y,step_size,score_dict,Loop_num,Run_name,cv_round,loop_dict,var_dict)) 
            round_data[cv_round] = loop_dict 
        loop_dict['AM Time Series Data'] = AM_EndOfLoopError
        loop_dict['Final Scores'] = score_dict
        """  AM Tuning is now Finished for the CV_split, Elastic Net is Next """
        EN = ElasticNetCV(l1_ratio=l1_ratios, cv=5, copy_X= True, normalize= True, random_state= 23)
        X_train = X.copy()
        X_train.replace(score_dict, inplace= True)
        y_train = y.copy()
        X_test = Test_set.iloc[:,:Test_set.shape[1]-1]
        X_test.replace(score_dict, inplace =True)
        X_test['Intercept']= 1
        y_test = pd.DataFrame(Test_set['pMeas'])
        EN.fit(X_train,y_train)
        y_pred = pd.DataFrame(EN.predict(X_test))
        Pearson_correlations.append(np.corrcoef(y_test.T,y_pred.T)[0][1])
        """Save Everything """
        ElasticNet_dict["y_pred"]= y_pred
        ElasticNet_dict['y_test']= y_test
        ElasticNet_dict['Alpha'] = EN.alpha_
        ElasticNet_dict['l1_ratio']= EN.l1_ratio_
        ElasticNet_dict['Parameters']= EN.get_params() 
        ElasticNet_dict["AlphaSpace"] = EN.alphas_
        loop_dict['ElasticNet'] = ElasticNet_dict
        round_data[cv_round]= loop_dict
    All_data[Run_name]= round_data
    np.save("All Data.npy",All_data)
    return np.mean(Pearson_correlations)

        