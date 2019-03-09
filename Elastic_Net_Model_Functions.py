# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:12:39 2019

@author: Roman
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics  import mean_squared_error as MSE 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import random



## Helper Functions Used in Main.py##
def CV_split(x,n):   #Function the splits shuffled data frame into n data sets. 
    x = shuffle(x)
    sets=[]
    for i in range(n):
        if i == 0:
            sets.append(x.iloc[i:int(round(len(x)/n)),:])
        else:
            sets = sets
            sets.append(x.iloc[(i*int(round(len(x)/n))+1):(i+1)*int(round(len(x)/n)),:])
    return sets



def exclude(lst, i):
    if i == 0:
        return lst[i+1:]

    return lst[:i] + lst[i+1:]

def PlotErrors():  ##Generates Plots Mid-AM Tuning ##
    time =np.arange(0,len(Errors))
    var_dict[variable] = [Errors,Scores]
    ax= plt.subplot(111)
    ax.plot(time,Errors, label='MSE')
    plt.title(str(Loop_num) +"_" + str(variable))
    plt.show()
        
def Amplitude_Tuning(X,y,step_size,score_dict,Loop_num,Run_name,cv_round,loop_dict,var_dict):
    for variable in score_dict.keys():
        Errors = []
        if Loop_num == 1:
            Scores = [random.randint(-10,10)]
            score_dict[variable] = Scores[-1]
            Errors.append(Get_Error(X,y,score_dict)) 
        else:
            Scores = [score_dict[variable]]
            Errors.append(Get_Error(X,y,score_dict))
        iteration = len(Errors)
        change_in_error= []
        gradient_direction = 1
        reversals = 0
        while (((Errors[-1]-Errors[0])/(Errors[0])) > -.05 and iteration <10000) and reversals < 3:
            perturbation = gradient_direction * step_size
            iteration = len(Errors)
            change_in_error.append((Errors[-1]-Errors[0])/(Errors[0]))
            second_deriv = pd.Series(change_in_error).pct_change()
            second_deriv = np.array(second_deriv)
            if iteration < 300:    
                message = ("Round_Loop:  "+str(cv_round)+"_"+ str(Loop_num)+ "   Iteration:  "+str(iteration)+  "   Variable:  " + str(variable)+"   Score:  "+str(Scores[-1])+"  Change in Error:  "+str((Errors[-1]-Errors[0])/(Errors[0])))
                print(message)   
                if change_in_error[-1] >= .02: ## If error gets 20 percent worse over time##
                    Scores.append(Scores[0])                   ## Reset back to square one##
                    score_dict[variable]= Scores[-1]
                    Errors.append(Get_Error(X,y,score_dict))
                    gradient_direction *= -1
                    reversals = reversals + 1
                else:                                           ## perturb score##
                    Scores.append(Scores[-1]+perturbation)
                    score_dict[variable] = Scores[-1]
                    Errors.append(Get_Error(X,y,score_dict))  
            else:
                if  (Errors[-1] - min(Errors))/min(Errors) > 0.03:
                    score_dict[variable] = Scores[Errors.index(min(Errors))]
                    print ("Curve Reversed")
                    break 
                elif abs(Errors[-1]-Errors[-50]) < 0.0001:
                    score_dict[variable] = Scores[Errors.index(min(Errors))]
                    print ("Curve is Flat")
                    break
                message = "Round_Loop:  "+str(cv_round)+"_"+ str(Loop_num)+ "   Iteration:  "+str(iteration)+  "   Variable:  " + str(variable)+"   Score:  "+str(Scores[-1])+"  Change in Error:  "+str((Errors[-1]-Errors[0])/(Errors[0]))
                print(message) 
                if change_in_error[-1] >= .02: ## If error gets 20 percent worse over time##
                    Scores.append(Scores[0])                   ## Reset back to square one##
                    score_dict[variable]= Scores[-1]
                    Errors.append(Get_Error(X,y,score_dict))
                    gradient_direction *= -1
                    reversals = reversals + 1 
                    print("GRADIENT REVERSAL"+ "  "+str(reversals))
                else:                                           ## perturb score##
                    Scores.append(Scores[-1]+perturbation)
                    score_dict[variable] = Scores[-1]
                    Errors.append(Get_Error(X,y,score_dict)) 
        PlotErrors()
    loop_dict[Loop_num]= var_dict    
    End_of_AM_Error= Get_Error(X,y,score_dict)
    return End_of_AM_Error

def Get_Error(X,y,score_dict):
     X_num = X.copy()
     X_num.replace(score_dict, inplace = True)
     X_train,X_test, y_train, y_test = train_test_split(X_num,y,test_size=.1, random_state=23)
     lin = LinearRegression()
     lin.fit(X_train,y_train)
     y_pred = lin.predict(X_test)
     Error = MSE(y_test,y_pred)
     return Error
    

    
    
    
    