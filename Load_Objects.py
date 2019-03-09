#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:48:15 2019

@author: romangutin
"""
import pandas as pd
import numpy as np
### LOAD FILES ###

All_data = np.load("All Data.npy").item() 







''' Variables '''

PseudoVSEPR_vars =['NH' , 'COO','NH3+','alphacarbon','c=o','1meth','2meth','3meth','Phenyl','Phenol','CH3','Iso','RTail']+ ['HTail']+ ['KTail'] + ['NegTail']+['STail']+['NTail']+['CTail']+["UTail"] +['WTail']+['PTail']
amino_letter = ['A','R','D','N','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
''' '''

''' Encodings '''
psuedo_VSEPR= pd.read_pickle('Terminus Encoding')           ## pseudo VSEPR encoded dataframe##
letter_based =  pd.read_pickle("Letter Based Encoding.pkl") ## Letter Based Encoding## 
Zero_Padding = pd.read_pickle("Zero_Padded_PseudoVSEPR")


''' Score Dictionaries '''
scores_pseudoVSEPR = np.load('Random Initialized Score Dictionary.npy').item() 
scores_letterbased = np.load("Randomly Intialized Scores for Letter-Based Encoding.npy").item() ##Random Letter Based Scores##
scores_ZeroPad = np.load("Randomly_Intitialized_Zero_Padding_Scores.npy").item()

