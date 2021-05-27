#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:04:03 2021

@author: aleksanderbielinski
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pandas_datareader import data,wb
'URTH'
symbols=['AEL','CRAI','RJF','SNX','TNK','UNM','005490.KS','007070.KS','2328.HK','3813.HK','0386.HK','BSE.AX','BAER.SW','CTT.LS','TCELL.IS','URTH']

#number of assets present, useful to have it in the form of variable
noa=len(symbols)

#designing a dataframe 

d1=pd.DataFrame()
for sym in symbols:
    d1[sym]=data.DataReader(sym,data_source='yahoo',start='2020-01-01', end='2021-01-01')['Adj Close']
d1.columns=symbols

#filling the null values

d1.fillna(method ='bfill', inplace = True) 

d1.fillna(method = 'ffill', inplace=True)

#Normalizing the data

fig, ax = plt.subplots()
(d1/d1.iloc[0]*100).plot(figsize=(12,10), lw=1.5, fontsize=12, ax=ax, grid=True)
for line in ax.get_lines():
    if line.get_label() == 'URTH':
        line.set_linewidth(4)
plt.title(label="Portfolio's Constituents Normalized Price for 01/01/2020-01/01/2021", size='xx-large')
plt.legend(loc="upper left",ncol=2) 
plt.xlabel("Date", size='x-large')
plt.show()


(d1/d1.iloc[0]*100).plot(figsize=(12,10))

#now for the logged returns

rets=np.log(d1/d1.shift(1))


#Getting the mean and var/cov of them for the specific time horizon excluding Sundays and Saturdays, weekday holidays and fixed data holidays 

meanrets=rets.mean()*252

varcov=rets.cov()*252

rets.var()*252


varcovar=rets.cov()*252

#correlation heatmap
corr=rets.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr,cmap='coolwarm', center=0, mask=mask, ax=ax, annot=True, linewidths=3, vmax=.3, square=True,)
plt.title(label="Correlations between Portfolio's Consituents", size='xx-large')

varcovar.to_excel("PMTF3.xlsx",sheet_name='VARCOV Matrix')  
corr.to_excel("bor1.xlsx",sheet_name='Correlation')  

#var covar heatmap

mask1 = np.triu(np.ones_like(varcovar, dtype=bool)) 

fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(varcovar,cmap='coolwarm', center=0, ax=ax, annot=True, cbar=True, linewidths=3, vmax=.3)
plt.title(label="Variance/Covariance Matrix for the Portfolio", size='xx-large')

r

#plotting a return series

rets.plot(figsize=(12,10), lw=1.5, fontsize=12, grid=True)
plt.title(label="Portfolio's Logaritmic Returns over 01/01/2020-01/01/2021 Period", size='xx-large')
plt.legend(loc="upper center",ncol=2) 
plt.xlabel("Date", size='xx-large')


#creating list of random weights

weights=np.random.random(noa)#random number form noa
weights/=np.sum(weights)# individually cannot be equal to sum 
print(weights)

#checking whether they sum to 1

weights.sum()

#calculating the returns for this weights np.dot in this case represents matrix multiplication

np.sum(rets.mean()* weights)*252 #expected portfolio return
np.dot(weights.T, np.dot(rets.cov() *252, weights))#expected variance
np.sqrt(np.dot(weights.T, np.dot(rets.cov() *252, weights)))#expected portfolio's standard deviation/volatility

#equally weighted portfolio

wei=(0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667,0.06666667)

#creating an array

weights1=np.array(wei)

#risk of the equally weighted portfiolo 
np.sum(rets.mean()* weights1)*252 #expected portfolio return
np.dot(weights1.T, np.dot(rets.cov() *252, weights1))#expected variance
np.sqrt(np.dot(weights1.T, np.dot(rets.cov() *252, weights1)))#expected portfolio's standard deviation/volatility

#Now for the array of returns and their standard deviation 

prets=[]
pvols=[]
for p in range(2500):
    weights=np.random.random(noa)
    weights/=np.sum(weights)
    prets.append(np.sum(rets.mean() *weights)*252)
    pvols.append(np.sqrt(np.dot(weights.T,
                                np.dot(rets.cov() *252, weights))))
    
prets=np.array(prets)
pvols=np.array(pvols)

#which can be used to obtain sharpe ratio

plt.figure(figsize=(12,6))
plt.scatter(pvols,prets, c=prets/pvols, marker='o',alpha=0.7)
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')

#Now we began to optimize the portfolio

def statistics(weights):
    weights=np.array(weights)
    pret=np.sum(rets.mean()*weights)*252
    pvol=np.sqrt(np.dot(weights.T, np.dot(rets.cov()*252,weights)))
    return np.array([pret,pvol,pret/pvol])

#return the negative of the sharpe ration for given weights (Maximising the Sharpe Ratio)

import scipy.optimize as sco

def min_func_sharpe(weights):
    return -statistics(weights)[2]

#we need a constraint

def sum_constraint(x):
    return np.sum(x)-1

#add a constraints
noa
cons=({'type':'eq','fun':sum_constraint}) #type equality constraints and a function is the one created above
bnds=tuple((0.02,1) for x in range(noa))#for each variable what is the larges possible value and the smallest possible vaule of a weight
bnds
#now optimize, produces a optimal weights for optimization, based on ealier creqadted ranom weights

opts=sco.minimize(min_func_sharpe, weights, method='SLSQP', bounds=bnds, constraints=cons)

#more accessible
opts['x'].round(3)

#now check the returns for each stock
statistics(opts['x']).round(3)

#now plot the model for the returns between 0 and 0.25

def min_func_port(weights):Part
    return statistics(weights)[1]

#possible return levels constraints

trets=np.linspace(0.0,0.4,300)#100 points between 0 and 0.5 equally spaced
tvols=[]

#fucntion for the constraints

def return_constraint(weights):
    return statistics(weights)[0]-tret

#now for the fucntion that dispalys retuns in the  established constraints

tvols=[]
for tret in trets:
    cons=({'type':'eq', 'fun':sum_constraint},
         {'type':'eq', 'fun': return_constraint})
    res=sco.minimize(min_func_port, weights, method='SLSQP', bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
    
#constructing a array

tvols=np.array(tvols)

#another optimizastion model that minimize the variance

def min_func_variance(weights):
    return statistics(weights)[1]**2

cons=({'type':'eq', 'fun':sum_constraint})
optv=sco.minimize(min_func_variance, weights, method='SLSQP', bounds=bnds, constraints=cons)
   
#more accessible way
optv['x'].round(3)
statistics(optv['x']).round(3)

#plotting an efficient frontier line
plt.figure(figsize=(12,7))
plt.title(label="Efficient Frontier",size='xx-large')
plt.scatter(pvols,prets,c=prets/pvols,marker='o')
#random portfolio composition
#plt.scatter(tvols,trets, c=trets/tvols, marker='x')
#efficient frontrier
plt.plot(tvols,trets,'b--')
#potfolio with higest sharpe ratio
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],'r*', markersize=15.0)
#min variance portfolio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],'y*', markersize=15.0)
plt.grid(True)
plt.xlabel('Expected Standard Deviation',size='x-large')
plt.ylabel('Expected Return',size='x-large')
plt.colorbar(label='Sharpe Ratio')