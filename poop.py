import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
# og is the orignal data#
og = pd.read_csv('bdata.20130222.mhci.txt', delimiter='\t')
# I only care about the sequences from humans#
human = og[og['species'] == 'human']
human['meas']= np.log10(human['meas'])
human['meas']= human['meas']*-1
human.rename(columns={'meas' :'pMeas'}, inplace=True)
j = []
for seq in human['sequence']:
    j.append(list(seq))
j= pd.DataFrame(j, index= human.index)
human = pd.concat([human,j],axis=1)
amino_letter = ['A','R','D','N','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
mol_weights =  [71.09,156.19,115.09,114.11,103.15,129.12,128.14,57.05,137.14,113.16,113.16,128.17,131.19,147.18,97.12,87.08,101.11,186.12,163.18,99.14]
def min_max(x,x_min,x_max):
    z = (2)*(x-x_min)/(x_max-x_min)-1
    return(z)
sc_weights = []
for i in mol_weights:
    sc_weights.append(min_max(i,np.min(mol_weights),np.max(mol_weights)))
ltw = dict(zip(amino_letter,sc_weights))
ltw_sorted = dict(sorted(ltw.items(), key=lambda kv: kv[1]))
list_length_allele = [] # list[peptidelength][allele I care about] = dataframe holding all data for a specific length and specific 
                            #allele.
dict_pep_length = {k:v for k,v in human.groupby('peptide_length')} #dictionary sectioning humans into df for each peptide length
for i in range(len(human['peptide_length'].unique())):
    list_length_allele.append({allele:val  for allele,val in dict_pep_length[human['peptide_length'].unique()[i]].groupby('mhc')})
nine_pep = list_length_allele[1][human['mhc'].unique()[1]]
nine_pep.replace(ltw_sorted,inplace=True)
nine_pep.to_csv('Length 9 Peptide Sequences Min-Max Scored by Mol Weight', sep=',')        
just_seq = pd.read_csv('just_sequence.csv')
#Building custom mexican hat wavelet#
r1= [-1/math.sqrt(6),2/math.sqrt(6),-1/math.sqrt(6),0,0,0,0,0,0]
r2=[0,0,0,-1/math.sqrt(6),2/math.sqrt(6),-1/math.sqrt(6),0,0,0]
r3= [0,0,0,0,0,0,-1/math.sqrt(6),2/math.sqrt(6),-1/math.sqrt(6)]
r4 = [-1/math.sqrt(18),-1/math.sqrt(18),1/math.sqrt(18),2/math.sqrt(18),2/math.sqrt(18),2/math.sqrt(18),-1/math.sqrt(18),-1/math.sqrt(18),-1/math.sqrt(18)]
r5= [1/math.sqrt(18),1/math.sqrt(18),1/math.sqrt(18),2/math.sqrt(18),2/math.sqrt(18),2/math.sqrt(18),1/math.sqrt(18),1/math.sqrt(18),1/math.sqrt(18)]
m_hat= pd.DataFrame([r1,r2,r3,r4,r5],index=['r1','r2','r3','r4','r5'],columns=[0,1,2,3,4,5,6,7,8])
just_seq.set_index('sequence',inplace=True)
W_seq = np.dot(just_seq,m_hat.T)
colnames = ['cd11','cd12','cd13','cd2','ca2']
W_seq =pd.DataFrame(W_seq, index=just_seq.index, columns = colnames)
nine_pep.set_index('sequence',inplace=True)
W_seq['pMeas']=nine_pep['pMeas']
#counting most frequent amino acid
h_al2 = human[human['mhc'] == human['mhc'].unique()[1]]
h_al2.head()
notltw_9 = h_al2[h_al2['peptide_length'] == 9]
i=[]
for seq in notltw_9['sequence']:
    i.append(list(seq))
i = pd.DataFrame(i, index= notltw_9.index)
notltw_9 = pd.concat([notltw_9,i], axis =1)
just_let = notltw_9.iloc[:,6:15]   #DATAFRAME WITH AMINO ACID LETTERS LISTED FOR 9LENGTH -ALELLE 2
for acid in amino_letter:
    sum((just_let == 'A').apply(np.count_nonzero)) 
count_dict = {}
for key in amino_letter:
    count_dict[key]=sum((just_let == key).apply(np.count_nonzero))
count_df = pd.DataFrame.from_dict(count_dict,orient='index',columns=['count'])
count_df.sort_values(by=['count'],inplace=True,ascending=False)
just_let.set_index(notltw_9['sequence'], inplace=True)       
