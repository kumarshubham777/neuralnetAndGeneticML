__author__ = 'Shubham'
import pandas as pd;
from sklearn.model_selection import train_test_split
from pandas import Series,DataFrame;
import numpy as np;
data1=[1,4.9,7,5]
arr1=np.array(data1)
#print(arr1)
#print(arr1.shape)
arr1=np.array([1,2,3],dtype=np.float64)
#print(arr1.dtype)

#print(arr2=='sam')
dataframe_all=pd.read_csv('dataset111.csv',header=None,sep=',')
fields=['DEFECT']
datacol=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fields)

datacol['DEFECT']=datacol['DEFECT'].replace(['yes','no'],[1,0])
y=datacol;
fieldsrow=['WMC','DIT','NOC','CBO','RFC','LCOM','Ca','Ce','NPM','LCOM3','LOC','DAM','MOA','MFA','CAM','IC','CBM','AMC']
datarow=pd.read_csv('telephonyandroiddataset.csv',skipinitialspace=True,usecols=fieldsrow)
x=datarow;
x['WMC']=x['WMC'].apply(lambda x:((x-1)/(213-1)))
x['DIT']=x['DIT'].apply(lambda x:((x-0)/(4-0)))
x['NOC']=x['NOC'].apply(lambda x:((x-0)/(4-0)))
x['CBO']=x['CBO'].apply(lambda x:((x-0)/(20-0)))
x['RFC']=x['RFC'].apply(lambda x:((x-2)/(214-2)))
x['LCOM']=x['LCOM'].apply(lambda x:((x-0)/(22578-0)))
x['Ca']=x['Ca'].apply(lambda x:((x-0)/(16-0)))
x['Ce']=x['Ce'].apply(lambda x:((x-0)/(17-0)))
x['NPM']=x['NPM'].apply(lambda x:((x-0)/(212-0)))
x['LCOM3']=x['LCOM3'].apply(lambda x:((x-1)/(2-1)))
x['LOC']=x['LOC'].apply(lambda x:((x-6)/(1100-6)))
x['DAM']=x['DAM'].apply(lambda x:((x-0)/(1-0)))
x['MOA']=x['MOA'].apply(lambda x:((x-0)/(37-0)))
x['AMC']=x['AMC'].apply(lambda x:((x-0)/(5-0)))

print(x)

# sigmoid function

def nonlin(x,deriv=False):
 if(deriv==True):
  return x*(1-x)
 return 1/(1+np.exp(-x))
np.random.seed(1)
syn0 = 2*np.random.random((18,1)) - 1


for iter in range(50000):
  l0 = x;
  l1 = nonlin(np.dot(l0,syn0))
  l1_error = y - l1;
  l1_delta = l1_error * nonlin(l1,True);
  syn0 += np.dot(l0.T,l1_delta);

print("Output After Training:")
print(l1)

