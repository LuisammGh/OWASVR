# -*- coding: utf-8 -*-
"""
Reference: Ordered Weighted Average Support Vector Regression
"""

import sys
import os
import pandas as pd
import numpy as np
#import cplex
import sklearn
from sklearn.model_selection import KFold
from sklearn.svm import SVR
import os.path
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from numpy import array

import gurobipy as gp
from gurobipy import GRB

def ifnull(pos, val):
    l = len(sys.argv)
    if len(sys.argv) <= 1 or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]

#inputs--------------------
#inputs--------------------
dataname=ifnull(1,"Quandt")
print(dataname)
nfold=int(ifnull(2,3))
print(nfold)
nvar=int(ifnull(3,1))
#-----------------------
DWdata= np.loadtxt('../data/'+dataname+'.txt', usecols=range(nvar+2))
ident=DWdata[:,0]
x1=DWdata[:,range(1,nvar+1)]
x2=DWdata[:,nvar+1]



# =============================================================================
#Data division

size=len(x1)
indices = np.arange(size)
indices_train, indices_test = train_test_split(indices, test_size=0.25, random_state=15) 
test_set=np.sort(indices_test) 
train_set=np.sort(indices_train)
print(train_set)
print(test_set)

x1_fin_train=x1[train_set]  
x2_fin_train=x2[train_set] 
x1_fin_test=x1[test_set]   
x2_fin_test=x2[test_set]   
all_train_fold=[]
all_test_fold=[]
x1_train=[]
x2_train=[]
x1_test=[]
x2_test=[]
kf = KFold(n_splits=nfold,shuffle=True, random_state=5)


for train_fold, test_fold in kf.split(indices_train): 
    all_train_fold.append(np.sort(indices_train[train_fold])) 
    x1_train.append(x1[np.sort(indices_train[train_fold])])   
    x2_train.append(x2[np.sort(indices_train[train_fold])].flatten()) 
    all_test_fold.append(np.sort(indices_train[test_fold]))  
    x1_test.append(x1[np.sort(indices_train[test_fold])]) 
    x2_test.append(x2[np.sort(indices_train[test_fold])].flatten()) 



# =============================================================================
#Parámetros, pesos y tiempo límite 
C=[0.01,0.1,1,10,100]
eps=[0,0.01,0.1,1,10]
sigma=[0.01,0.1,1,10,100]
pesos=[]
pesos.append("SUM")
pesos.append("MAX")
pesos.append("kC") 
pesos.append("akC")
pesos.append("MEDIAN")
pesos.append("Trimmed-mean")
timelim=900



#=============================================================================
#auxiliar
def auxiliar(TipoPeso,x,y,x_test,y_test,C,epsilon,sigma,nvar,a_val,a2_val):
  #--------PESOS LAMBDA------------------------------------------------------#
   n=len(x)
   if TipoPeso==0:
   #SUM
       l=[]
       for j in range(0,n):
           l.append(1)
   if TipoPeso==1:
   #MAX
       l=[]
       for j in range(0,n-1):
           l.append(0)
       l.append(1)
   #kC
   if TipoPeso==2:
       l=[]
       mitad=int(np.floor(n/2))
       print(n/2)
       for j in range(0,mitad):
           l.append(0)
       for j in range(mitad,n):
           l.append(1)

       for j in range(mitad+1,n):
           l.append(1)
#---------------------------------------------------------------------------#
   m = gp.Model("OWA_auxiliar")
   m.setParam('TimeLimit', timelim)
    

   b_var=m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name="b")
   theta_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="theta")
   z_vars=m.addVars(range(0,n),range(0,n),vtype=GRB.BINARY,name="z")
   xi_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi")
   xi2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi2")
   z_auxi=m.addVars(range(0,n),range(0,n),vtype=GRB.CONTINUOUS,lb=0,ub=1,name="z_auxi")

   objective=C*gp.quicksum(l[i]*theta_vars[i] for i in range(0,n))

   m.addConstrs(gp.quicksum(z_vars[i,j] for i in range(0,n))==1 for j in range(0,n))
   m.addConstrs(gp.quicksum(z_vars[i,j] for j in range(0,n))==1 for i in range(0,n))


   m.addConstrs(y[i]-gp.quicksum((a_val[j]-a2_val[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for j in range(0,n))-b_var<=epsilon+xi_vars[i] for i in range(0,n))

   m.addConstrs(-y[i]+gp.quicksum((a_val[j]-a2_val[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2)))for j in range(0,n))+b_var<=epsilon+xi2_vars[i] for i in range(0,n))
    
    
   m.addConstrs(z_auxi[i,k]==gp.quicksum(z_vars[i,j] for j in range(0,k)) for i in range(0,n) for k in range(0,n))
    #M=4
   # m.addConstrs(theta_vars[k]>=xi_vars[i]+xi2_vars[i]-M*(1-gp.quicksum(z_vars[i,j] for j in range(0,k))) for i in range(0,n) for k in range(0,n))
   for i in range(0,n):
       for k in range(0,n):
           m.addGenConstrIndicator(z_auxi[i,k],1,theta_vars[k]-xi_vars[i]-xi2_vars[i]>=0)
    

   tsol1=time.time()
   m.params.NonConvex = 2
   m.setObjective(objective, GRB.MINIMIZE)

   m.optimize()
   tsol2=time.time()

   obj=m.ObjVal
   INTERCEPT=b_var.X 
   status=m.status    
   nn=len(x_test)
    
   m.reset()
   stime=tsol2-tsol1
   return(obj,stime, status, INTERCEPT)


#------------------------------------------------------------------------------

#PARA PESOS MONOTONOS#
def owa_svr_mon_kernel(TipoPeso,x,y,x_test,y_test,C,epsilon,sigma,nvar):
  #--------PESOS LAMBDA------------------------------------------------------#
    n=len(x)
    if TipoPeso==0:
    #SUM
        l=[]
        for j in range(0,n):
            l.append(1)
    if TipoPeso==1:
    #MAX
        l=[]
        for j in range(0,n-1):
            l.append(0)
        l.append(1)
    #kC
    if TipoPeso==2:
        l=[]
        mitad=int(np.floor(n/2))
        print(n/2)
        for j in range(0,mitad):
            l.append(0)
        for j in range(mitad,n):
            l.append(1)

        for j in range(mitad+1,n):
            l.append(1)
#---------------------------------------------------------------------------#
    m = gp.Model("OWA_SVR_MON_KERNEL")
    m.setParam('TimeLimit', timelim)
    
    a_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="a")
    a2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="a2")
    beta_vars=m.addVars(range(0,n),range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="beta")


    objective=-1/2*gp.quicksum((a_vars[i]-a2_vars[i])*(a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for i in range(0,n) for j in range(0,n))

    objective-=epsilon*gp.quicksum(a_vars[i]+a2_vars[i] for i in range(0,n))
    objective+=gp.quicksum(y[i]*(a_vars[i]-a2_vars[i]) for i in range(0,n))


    m.addConstr(gp.quicksum(a_vars[i]-a2_vars[i] for i in range(0,n))==0)
    m.addConstrs(gp.quicksum(beta_vars[i,j]*C*l[j] for j in range(0,n))>=a_vars[i] for i in range(0,n))
    m.addConstrs(gp.quicksum(beta_vars[i,j]*C*l[j] for j in range(0,n))>=a2_vars[i] for i in range(0,n))

    m.addConstrs(gp.quicksum(beta_vars[i,j] for i in range(0,n))==1 for j in range(0,n))
    m.addConstrs(gp.quicksum(beta_vars[i,j] for j in range(0,n))==1 for i in range(0,n))


    tsol1=time.time()
    m.setObjective(objective, GRB.MAXIMIZE)
    m.params.NonConvex = 2
    m.optimize()
    tsol2=time.time()
    stime=tsol2-tsol1
    obj=m.ObjVal
    

    status=m.status    
    nn=len(x_test)
    aux=0
    for i in range(0,n):
        if a_vars[i].X>0.001 and a_vars[i].X<sum(beta_vars[i,j].X*C*l[j] for i in range(0,n) for j in range(0,n)):
            INTERCEPT=y[i]-sum((a_vars[j].X-a2_vars[j].X)*np.exp(-np.dot(x[j]-x[i],x[j]-x[i])/(2*pow(sigma,2)))for j in range(0,n))-epsilon   
            aux=1
            break
        elif a2_vars[i].X>0.001 and a2_vars[i].X<sum(beta_vars[i,j].X*C*l[j] for i in range(0,n) for j in range(0,n)):
            INTERCEPT=y[i]-sum((a_vars[j].X-a2_vars[j].X)*np.exp(-np.dot(x[j]-x[i],x[j]-x[i])/(2*pow(sigma,2)))for j in range(0,n))+epsilon 
            aux=1
            break
    
    a_sol=[a_vars[i].X for i in range(0,n)]
    a2_sol=[a2_vars[i].X for i in range(0,n)]
    m.reset()
    auxtime=0
    
    if aux<0.5:
        results=auxiliar(TipoPeso,x,y,x_test,y_test,C,epsilon,sigma,nvar,a_sol,a2_sol)
        INTERCEPT=results[3]
        auxtime=results[1]

    y_pred=[INTERCEPT+sum((a_sol[i]-a2_sol[i])*np.exp(-np.dot(x[i]-x_test[j],x[i]-x_test[j])/(2*pow(sigma,2)))for i in range(0,n)) for j in range(0,nn)]

    mae=mean_absolute_error(y_test, y_pred)
    mse=mean_squared_error(y_test,y_pred)
    return(obj,INTERCEPT,stime,status,mae,mse,aux,auxtime,y_test,y_pred)
#=================================================================================================================================================#

   
results0="results_K_OWA_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results0)==False):
    f0 = open(results0, "a")
    f0.write(f'Data \t Model \t Weight \t C \t eps\t sigma\t fold \t size \t obj \t intercept\t  \t time \t status \t MAE \t MSE  \n')
    f0.close()

    
results1="Best_Mae.txt"
if(os.path.isfile(results1)==False):
    f1= open(results1, "a")
    f1.write(f' Data \t Model \t  Weight \t C \t eps\t sigma \t size \t obj \t intercept \t time \t status \t MAE \t MSE  \n')      
    f1.close()
    
results2="Best_Mse.txt"
if(os.path.isfile(results2)==False):
    f2= open(results2, "a")
    f2.write(f'Data \t Model \t  Weight \t C \t eps \t sigma\t size \t obj \t intercept \t time \t status  \t MAE \t MSE \n')
    f2.close()



results3="y_test_KOWASVR_Fold_"+dataname+".txt"
if(os.path.isfile(results3)==False):
    f3= open(results3, "a")
    f3.write(f'Data\t  Model \t Weight\t C\t eps\t fold  \t y_test \n')                    
    f3.close()

results4="y_pred_KOWASVR_Fold_"+dataname+".txt"
if(os.path.isfile(results4)==False):
    f4= open(results4, "a")
    f4.write(f'Data\t Model \t  Weight\t C\t eps\t fold  \t y_pred \n')                    
    f4.close()
    
results5="y_error_KOWASVR_Fold_"+dataname+".txt"
if(os.path.isfile(results5)==False):
    f5= open(results5, "a")
    f5.write(f'Data\t Model \t  Weight\t C\t eps\t fold  \t error \n')                    
    f5.close()
            
for k in range(0,3):
    measures_results=[]
    for ii in range(0,len(C)):
        for j in range(0,len(eps)):
            for m in range(0,len(sigma)): #SIGMA
                times=[]
                MAE=[]
                MSE=[]
                for i in range(0,nfold):
                    timelim=900
                    results=owa_svr_mon_kernel(k,x1_train[i],x2_train[i],x1_test[i],x2_test[i],C[ii],eps[j],sigma[m],nvar)
                    f0 = open(results0, "a")
                    f0.write(f' {dataname}\t K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t {sigma[m]}\t{i}\t {len(all_train_fold[i])}\t {results[0]:.4f}\t {results[1]:.4f}\t \
                             {results[2]+results[7]:.4f}\t {results[3]:.4f}\t {results[4]:.4f}\t {results[5]:.4f} \n')   
                    f0.close()  
                    
                    f3= open(results3, "a")
                    f3.write(f'{dataname}\t K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                    for ll in results[8]:
                        f3.write(f'\t {ll:.4f}')
                    f3.write(f'\n')
                    f3.close() 
                    
                    f4= open(results4, "a")
                    f4.write(f'{dataname}\t K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                    for ll in results[9]:
                        f4.write(f'\t {ll:.4f}')
                    f4.write(f'\n')
                    f4.close()  
                    
                    
                    er=results[8]-results[9]
                    
                    f5= open(results5, "a")
                    f5.write(f'{dataname}\t  K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                    for ll in er:
                        f5.write(f'\t {ll:.4f}')
                    f5.write(f'\n')
                    f5.close()  
                    times.append(results[2]+results[7])
                    MAE.append(results[4])
                    MSE.append(results[5])
                measures_results.append([C[ii],eps[j],sigma[m],np.mean(times),np.mean(MAE),np.mean(MSE)])
    best_MAE=[np.argmin(i) for i in zip(*measures_results)][4]
    best_MSE=[np.argmin(i) for i in zip(*measures_results)][5]
    
    timelim=1800
    results=owa_svr_mon_kernel(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MAE][0],measures_results[best_MAE][1],measures_results[best_MAE][2],nvar)
    f1 = open(results1, "a")
    f1.write(f'{dataname}\t {pesos[k]}\t {measures_results[best_MAE][0]}\t {measures_results[best_MAE][1]}\t {measures_results[best_MAE][2]}\t{len(x1_fin_train)}\t {results[0]:.4f}\t {results[1]:.4f}\t \
              {results[2]+results[7]:.4f}\t {results[3]:.4f} \t {results[4]:.4f} \t {results[5]:.4f}  \n')   
    f1.close()   
    
  
    timelim=1800
    results=owa_svr_mon_kernel(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MSE][0],measures_results[best_MSE][1],measures_results[best_MSE][2],nvar)
    f2 = open(results2, "a")
    f2.write(f'{dataname}\t {pesos[k]}\t {measures_results[best_MSE][0]}\t {measures_results[best_MSE][1]}\t {measures_results[best_MSE][2]}\t {len(x1_fin_train)}\t {results[0]:.4f}\t {results[1]:.4f}\t \
              {results[2]+results[7]:.4f}\t {results[3]:.4f} \t {results[4]:.4f} \t {results[5]:.4f} \n')   
    f2.close()      
    
   
    

   