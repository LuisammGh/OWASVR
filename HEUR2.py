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
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import d2_absolute_error_score
from numpy import array

import gurobipy as gp
from gurobipy import GRB

def ifnull(pos, val):
    l = len(sys.argv)
    if len(sys.argv) <= 1 or (sys.argv[pos] == None):
        return val
    return sys.argv[pos]

#inputs--------------------
dataname=ifnull(1,"Quandt") #Data
print(dataname)
nfold=int(ifnull(2,3)) #Number of folds
print(nfold)
nvar=int(ifnull(3,1)) #Number of variables
#-----------------------
DWdata= np.loadtxt('../data/'+dataname+'.txt', usecols=range(nvar+2))
ident=DWdata[:,0]
x1=DWdata[:,range(1,nvar+1)]
x2=DWdata[:,nvar+1]



# =============================================================================
#Data division
size=len(x1)
indices = np.arange(size)
indices_train, indices_test = train_test_split(indices, test_size=0.25, random_state=15) #indices_train contains 75% of the data and indices_test contains 25% of the data
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
# A tener en cuenta, cambiar el número de splits
kf = KFold(n_splits=nfold,shuffle=True, random_state=5)


for train_fold, test_fold in kf.split(indices_train): 
    all_train_fold.append(np.sort(indices_train[train_fold])) 
    x1_train.append(x1[np.sort(indices_train[train_fold])])  
    x2_train.append(x2[np.sort(indices_train[train_fold])].flatten()) 
    all_test_fold.append(np.sort(indices_train[test_fold]))  
    x1_test.append(x1[np.sort(indices_train[test_fold])]) 
    x2_test.append(x2[np.sort(indices_train[test_fold])].flatten()) 


# =============================================================================
#Parameters, weights and time limits
C=[0.01,0.1,1,10,100]
eps=[0,0.01,0.1,1,10]
pesos=[]
pesos.append("SUM")
pesos.append("MAX")
pesos.append("kC") 
pesos.append("akC")
pesos.append("MEDIAN")
pesos.append("Trimmed-mean")
timelim=900


# =============================================================================

def HEUR2(TipoPeso,x,y,x_test,y_test,C,epsilon,nvar):
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
   if TipoPeso==3:
        # AkC
        l=[]
        mitad=int(np.floor(n/2))
        for j in range(0,mitad):
            l.append(1)
        for j in range(mitad,n):
            l.append(0)

    #MEDIAN
   if TipoPeso==4:
        l=[]
        mitad=int(np.floor(n/2))
        for j in range(0,mitad):
            l.append(0)
        l.append(1)
        for j in range(mitad+1,n):
            l.append(0)
            
   if TipoPeso==5:
        #KCENTRUM
        l=[]
        inf=int(np.floor(0.25*n))
        sup=int(np.floor(0.75*n))
        for i in range(0,inf):
             l.append(0)
        for i in range(inf,sup):
             l.append(1)
        for i in range(sup,n):
             l.append(0)
#---------------------------------------------------------------------------#
   svr=SVR(kernel='linear',C=C,epsilon=epsilon)
   svr.fit(x,y)
   pred=svr.predict(x)
    #svr.fit(x.reshape(-1,1),y)
    #pred=svr.predict(x.reshape(-1,1))

   errores=[abs(pred[i]-y[i]) for i in range(0,n)]
   argerrores=np.argsort(errores)

   m = gp.Model("HEUR2")
   m.setParam('TimeLimit', timelim)

   w_vars=m.addVars(range(0,nvar),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name="w")
   b_var=m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name="b")
   xi_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi")
   xi2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi2")
   objective=1/2*gp.quicksum(w_vars[i]*w_vars[i] for i in range(0,nvar))+C*gp.quicksum(l[i]*(xi_vars[argerrores[i]]+xi2_vars[argerrores[i]]) for i in range(0,n))

   m.addConstrs(xi_vars[i]+epsilon>=y[i]-b_var-gp.quicksum(w_vars[j]*x[i][j] for j in range(0,nvar)) for i in range(0,n))
   m.addConstrs(xi2_vars[i]+epsilon>=-y[i]+b_var+gp.quicksum(w_vars[j]*x[i][j] for j in range(0,nvar)) for i in range(0,n))
    

   tsol1=time.time()

   m.setObjective(objective, GRB.MINIMIZE)
   m.optimize()
   tsol2=time.time()

   status=m.status    
   INTERCEPT=b_var.X
   SLOPE=[w_vars[j].X for j in range(0,nvar)]
   obj=m.ObjVal
   m.reset()
   nn=len(x_test)

    
    #PARA CALCULAR EL ANCHO DE BANDA
   nn=len(x_test)
   y_pred=[INTERCEPT+np.dot(SLOPE,x_test[i]) for i in range(0,nn)]
   mae=mean_absolute_error(y_test, y_pred)
   mse=mean_squared_error(y_test,y_pred)
    
   return(obj,INTERCEPT,SLOPE,tsol2-tsol1,status,mae,mse,y_test,y_pred)
# =============================================================================


# =============================================================================
#



   
results0="results_HEUR2_Fold_"+dataname+".txt"
if(os.path.isfile(results0)==False):
    f0 = open(results0, "a")
    f0.write(f'Data\t Model\t Weight\t C\t eps\t fold \t size \t obj \t intercept\t slope\t time \t status  \t MAE \t MSE  \n')
    f0.close()
    
results1="Best_Mae.txt"
if(os.path.isfile(results1)==False):
    f2= open(results1, "a")
    f2.write(f'Data\t Model\t Weight\t C\t eps\t size \t obj \t intercept\t slope\t time \t status \t MAE \t MSE   \n')                    
    f2.close()
    
results2="Best_Mse.txt"
if(os.path.isfile(results2)==False):
    f3= open(results2, "a")
    f3.write(f'Data\t Model\t Weight\t C\t eps\t fold \t obj \t intercept\t slope\t time \t status  \t MAD \t MSE  \n')                   
    f3.close()


results3="y_test_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results3)==False):
    f11= open(results3, "a")
    f11.write(f'Data\t Model\t Weight\t C\t eps\t fold  \t y_test \n')                    
    f11.close()

results4="y_pred_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results4)==False):
    f12= open(results4, "a")
    f12.write(f'Data\t Model\t Weight\t C\t eps\t fold  \t y_pred \n')                    
    f12.close()
    
results5="y_error_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results5)==False):
    f13= open(results5, "a")
    f13.write(f'Data\t Model\t Weight\t C\t eps\t fold  \t error \n')                    
    f13.close()
   

for k in range(0,6):
    measures_results=[]
    for ii in range(0,len(C)):
        for j in range(0,len(eps)):
            ancho=[]
            times=[]
            MAD=[]
            MSE=[]
            for i in range(0,nfold):
                results=HEUR2(k,x1_train[i],x2_train[i],x1_test[i],x2_test[i],C[ii],eps[j],nvar)
                f0= open(results0, "a")
                f0.write(f'{dataname}\t HEUR2\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}\t {len(all_train_fold[i])}\t {results[0]:.4f}\t {results[1]:.4f}\t\
                         {[round(results[2][i],4) for i in range(0,nvar)]}\t \
                          {results[3]:.4f}\t {results[4]:.4f} \t {results[5]:.4f} \t {results[6]:.4f}   \n')   
                f0.close()   
                f3= open(results3, "a")
                f3.write(f'{dataname}\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                for ll in results[7]:
                    f3.write(f'\t {ll:.4f}')
                f3.write(f'\n')
                f3.close() 
                
                f4= open(results4, "a")
                f4.write(f'{dataname}\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                for ll in results[8]:
                    f4.write(f'\t {ll:.4f}')
                f4.write(f'\n')
                f4.close()  
                
                
                er=results[7]-results[8]
                
                f5= open(results5, "a")
                f5.write(f'{dataname}\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                for ll in er:
                    f5.write(f'\t {ll:.4f}')
                f5.write(f'\n')
                f5.close()  
                
                times.append(results[3])
                MAD.append(results[5])
                MSE.append(results[6])
            measures_results.append([C[ii],eps[j],np.mean(times),np.mean(MAD),np.mean(MSE)])
    best_MAE=[np.argmin(i) for i in zip(*measures_results)][3]
    best_MSE=[np.argmin(i) for i in zip(*measures_results)][4]
   
    

    results=HEUR2(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MAE][0],measures_results[best_MAE][1],nvar)
    f1= open(results1, "a")
    f1.write(f'{dataname}\t HEUR2\t {pesos[k]}\t {measures_results[best_MAE][0]}\t {measures_results[best_MAE][1]}\t {len(x1_fin_train)}\t {results[0]:.4f}\t {results[1]:.4f}\t {[round(results[2][i],4) for i in range(0,nvar)]}\t \
              {results[3]:.4f}\t {results[4]:.4f} \t {results[5]:.4f} \t {results[6]:.4f}   \n')   
    f1.close()   
    
    
    results=HEUR2(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MSE][0],measures_results[best_MSE][1],nvar)
    f2 = open(results2, "a")
    f2.write(f'{dataname}\t HEUR2 \t {pesos[k]}\t {measures_results[best_MSE][0]}\t {measures_results[best_MSE][1]}\t {len(x1_fin_train)}\t {results[0]:.4f}\t {results[1]:.4f}\t {[round(results[2][i],4) for i in range(0,nvar)]}\t \
              {results[3]:.4f}\t {results[4]:.4f} \t {results[5]:.4f} \t {results[6]:.4f}  \n')   
    f2.close()     
    
   
    

   