# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:56:44 2018

@author: RomanGutin
"""
import pandas as pd
import numpy as np
import random
#Frequency Tuning Loop OLD##
amino_letter = ['A','R','D','N','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
length_scores =[4,8,6,6,5,7,7,4,7,5,6,8,7,8,5,5,5,9,8,5] 
FM_df = pd.DataFrame(0, index= just_let.index, columns= range(0,81))
FM_score_dict = dict(zip(amino_letter,length_scores))

#splitting amino letter into new independent variables based on its length score#
fm_letter_dict ={}
for letter in amino_letter:
    new_vars =[]
    for i in range(FM_score_dict[letter]):
        new_vars.append(letter+str(i+1))
    fm_letter_dict[letter]=new_vars
        
#generate new FM_tuned dataframe        
for seq in FM_df.index:
    letter_list= list(seq) 
    for letter in letter_list: 
        for var in fm_letter_dict[letter]:
            row= FM_df.loc[seq,:]
            spot= row[row==0].index[0]
            FM_df.loc[seq,spot]= var


FM_df= pd.read_csv('Frequency Tuned Dataset') #data after frequency tuning wit
FM_df.set_index('sequence', inplace= True)
FM_df_arr = np.array(FM_df.values, dtype=[('O', np.float)]).astype(np.float)


  #New letter to weight holding the new FM tuned variables
ltw_fm_MLE={}
for amino in amino_letter:
    for var in fm_letter_dict[amino]:
        ltw_fm_MLE[var]= ltw_AM_n[amino]
     

ltw_fm_MLE = np.load('ltw_fm_MLE.npy').item() 
##FM Conversion Loop Easiest Encoding##
fm_vars =['NH' , 'COO','NH3+','alphacarbon','c=o','1meth','2meth','3meth','Phenyl','Phenol','CH3','Iso','RTail']+ ['MTail']+['HTail']+ ['KTail'] + ['NegTail']+['COH']+['NTail']+['CTail']+["UTail"] +['WTail']+['PTail']
am_scores = random.sample(range(-50,50),len(fm_vars))   # construct score dictionary with these #

## Side Chains ##
R = ['2meth']+ ['RTail']
H = ['HTail']
K = ['3meth']+ ['KTail']
D = ['NegTail']
E = ['1meth'] +['NegTail']
S = ['COH']
T = S +['CH3']
N = ['NTail']
Q= ['1meth'] + ['NTail']
C = ['CTail']
U = ["UTail"]
G = []
P =['PTail']
A =  ['CH3']
V =['Iso']
I = V + ['CH3']
L = ['1meth'] + ['Iso']
M = ['MTail']
F =  ['Phenyl']
Y =  ['Phenol']
W = ['WTail']
score_dict = dict(zip(fm_vars,am_scores))
amino_notstringlist = [R,H,K,D,E,S,T,N,Q,C,U,G,P,A,V,I,L,M,Y,W,F]
strings =['R','H','K','D','E','S','T','N','Q','C','U','G','P','A','V','I','L','M','Y','W','F']
side_chain = dict(zip(strings,amino_notstringlist))

### Terminus Backbone Encoding ###
N_term = ['NH3+','alphacarbon','c=o']
Inter = ['NH','alphacarbon', 'c=o']
C_term = ['NH','alphacarbon', 'COO']

# Constructing the Dataframe#
def ZeroPad():
    while len(letter_encoding) < 4:
        letter_encoding.append(0)
        
rows = []
for seq in just_let.index:
    letter_list = list(seq)
    seq_encoding = []
    for index in range(len(letter_list)):
        if index == 0:
           letter_encoding = N_term[0:2] + side_chain[letter_list[index]]
           ZeroPad()
           letter_encoding += [N_term[-1]]
        elif index == 8:
            letter_encoding = C_term[0:2]+ side_chain[letter_list[index]]
            ZeroPad()
            letter_encoding += [C_term[-1]]
        else:
            letter_encoding = Inter[0:2]+ side_chain[letter_list[index]] 
            ZeroPad()
            letter_encoding += [Inter[-1]]
        seq_encoding += letter_encoding        
    rows.append(seq_encoding)
rows = np.array(rows)
Padded_Encoding = pd.DataFrame(rows, index = just_let.index)           
            
