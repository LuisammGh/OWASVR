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
#inputs--------------------
dataname=ifnull(1,"Quandt") #Data 
print(dataname)
nfold=int(ifnull(2,3)) #Number of folds
print(nfold)
nvar=int(ifnull(3,1)) #Number of independent variables
#-----------------------
DWdata= np.loadtxt('../data/'+dataname+'.txt', usecols=range(nvar+2))
ident=DWdata[:,0]
x1=DWdata[:,range(1,nvar+1)]
x2=DWdata[:,nvar+1]



# =============================================================================
#Data division
size=len(x1)
indices = np.arange(size)
#print(indices)
indices_train, indices_test = train_test_split(indices, test_size=0.25, random_state=15) #indices_train includes 75% of the data and indices_test contains 25% of the data
test_set=np.sort(indices_test) #Test
train_set=np.sort(indices_train)#Train
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


for train_fold, test_fold in kf.split(indices_train): #The elements will be indices_train[train_fold]/indices_train[test_fold]
    all_train_fold.append(np.sort(indices_train[train_fold])) #elements of the train in that fold
    x1_train.append(x1[np.sort(indices_train[train_fold])])   #x1 of the element of the train in that fold
    x2_train.append(x2[np.sort(indices_train[train_fold])].flatten()) #x2 of the elements of the train in that fold
    all_test_fold.append(np.sort(indices_train[test_fold]))  #elements of the test in that fold
    x1_test.append(x1[np.sort(indices_train[test_fold])]) #x1 of the elements of the test in that fold
    x2_test.append(x2[np.sort(indices_train[test_fold])].flatten()) #x2 of the elements of the test in that fold

    
 

# =============================================================================
#Parameters, time and weights definition
C=[0.01,0.1,1,10,100]
eps=[0,0.01,0.1,1,10]
pesos=[]
pesos.append("SUM")
pesos.append("MAX")
pesos.append("kC") 
pesos.append("akC")
pesos.append("MEDIAN")
pesos.append("Trimmed-mean")
timelim=1800


# =============================================================================

def HEUR1(TipoPeso,x,y,x_test,y_test,C,epsilon,nvar):
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
            
    m = gp.Model("OWA_Flores_Sosa")

    a_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,ub=C,name="a")
    a2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,ub=C,name="a2")
    
    objective=-1/2*gp.quicksum((a_vars[i]-a2_vars[i])*(a_vars[j]-a2_vars[j])*np.dot(x[i],x[j]) for i in range(0,n) for j in range(0,n))-epsilon*gp.quicksum(a_vars[i]     +a2_vars[i] for i in range(0,n))+gp.quicksum(y[i]*(a_vars[i]-a2_vars[i]) for i in range(0,n))

    m.addConstr(gp.quicksum(a_vars[i]-a2_vars[i] for i in range(0,n))==0)
    
    tsol1=time.time()
    m.setObjective(objective, GRB.MAXIMIZE)
    m.setParam('Threads',4)
    m.optimize()
    tsol2=time.time()
    status=m.status    
    a_sol=[]
    a2_sol=[]
    for i in range(0,n):
        a_sol.append(a_vars[i].X)
        a2_sol.append(a2_vars[i].X)
     
    w_orig=[sum((a_sol[i]-a2_sol[i])*x[i][j] for i in range(0,n)) for j in range(0,nvar)]
    
    w_matrix=[[(a_sol[i]-a2_sol[i])*x[i][j] for i in range(0,n)] for j in range(0,nvar)]
    
    w_vectors=np.sort(w_matrix)

    SLOPE=[sum(l[i]*w_vectors[j][i] for i in range(0,n)) for j in range(0,nvar)]
    print(SLOPE)
    
    m.reset()
        
    bindex=-10
    bu1=-100
    bl1=-100
    for i in range(0,n):
        if a_sol[i]>0.001 and a_sol[i]<C-0.001:
            INTERCEPT=-epsilon+y[i]-np.dot(SLOPE,x[i])
            bindex=i
    if bindex<-1:
        for i in range(0,n):
            if a2_sol[i]>0.001 and a2_sol[i]<C-0.001:
                INTERCEPT=epsilon+y[i]-np.dot(SLOPE,x[i])
                bindex=i
    if bindex<-1:        
        inferiores=[]
        index_inf=[]
        index_sup=[]
        superiores=[]
        for i in range(0,n):
            if a_sol[i]<C-0.001 or a2_sol[i]>0.001:
                inferiores.append(-epsilon+y[i]-np.dot(w_orig,x[i]))
                index_inf.append(i)
            if a_sol[i]>0+0.001 or a2_sol[i]<C-0.001:
                superiores.append(-epsilon+y[i]-np.dot(w_orig,x[i]))
                index_sup.append(i)
        bl=index_inf[np.argmax(inferiores)]
        bu=index_sup[np.argmin(superiores)]
        bl1=(-epsilon+y[bl]-np.dot(SLOPE,x[bl]))
        bu1=(-epsilon+y[bu]-np.dot(SLOPE,x[bu]))
        INTERCEPT=(bl1+bu1)/2       



    nn=len(x_test)
    erroresl2=[(abs(-np.dot(SLOPE,x_test[i])+y_test[i]-INTERCEPT))/np.sqrt(1+np.dot(SLOPE,SLOPE)) for i in range(0,nn)]
    argerroresl2=np.argsort(erroresl2)
    y_pred=[INTERCEPT+np.dot(SLOPE,x_test[i]) for i in range(0,nn)]
    mae=mean_absolute_error(y_test, y_pred)
    mse=mean_squared_error(y_test,y_pred)
    stime=tsol2-tsol1
    
    return(INTERCEPT,SLOPE,stime,mae,mse,y_test,y_pred)
# =============================================================================



   
results0="results_owa_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results0)==False):
    f0 = open(results0, "a")
    f0.write(f'Data\t Model\t Weight\t C\t eps\t fold \t size \t intercept\t slope\t time \t MAE \t MSE \n')
    f0.close()

    
results1="Best_Mae.txt"
if(os.path.isfile(results1)==False):
    f2= open(results1, "a")
    f2.write(f'Data\t Model\t Weight\t C\t eps\t size \t intercept\t slope\t time \t MAE \t MSE \n')                    
    f2.close()
    
results2="Best_Mse.txt"
if(os.path.isfile(results2)==False):
    f3= open(results2, "a")
    f3.write(f'Data\t Model\t Weight\t C\t eps\t size \t obj \t intercept\t slope\t time \t status  \t MAD \t MSE \n')                   
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
            MAE=[]
            MSE=[]
            for i in range(0,nfold):
                results=HEUR1(k,x1_train[i],x2_train[i],x1_test[i],x2_test[i],C[ii],eps[j],nvar)
                f0= open(results0, "a")
                f0.write(f'{dataname}\t HEUR1\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}\t {len(all_train_fold[i])}\t {results[0]:.4f}\t {[round(results[1][i],4) for i in range(0,nvar)]}\t {results[2]:.4f} \t {results[3]:.4f} \t {results[4]:.4f}\n')   
                f0.close() 
                
                f11= open(results3, "a")
                f11.write(f'{dataname}\t HEUR1\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                for ll in results[5]:
                    f11.write(f'\t {ll:.4f}')
                f11.write(f'\n')
                f11.close() 
                
                f12= open(results4, "a")
                f12.write(f'{dataname}\t HEUR1\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                for ll in results[6]:
                    f12.write(f'\t {ll:.4f}')
                f12.write(f'\n')
                f12.close()  
                
                
                er=results[5]-results[6]
                
                f13= open(results5, "a")
                f13.write(f'{dataname}\t HEUR1\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                for ll in er:
                    f13.write(f'\t {ll:.4f}')
                f13.write(f'\n')
                f13.close()  
                
                times.append(results[2])
                MAE.append(results[3])
                MSE.append(results[4])
            measures_results.append([C[ii],eps[j],np.mean(times),np.mean(MAE),np.mean(MSE)])
    #POR AQUÍ
    best_MAE=[np.argmin(i) for i in zip(*measures_results)][3]
    best_MSE=[np.argmin(i) for i in zip(*measures_results)][4]
    
    
  

    results=HEUR1(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MAE][0],measures_results[best_MAE][1],nvar)
    f1 = open(results1, "a")
    f1.write(f'{dataname}\t HEUR1\t {pesos[k]}\t {measures_results[best_MAE][0]}\t {measures_results[best_MAE][1]}\t {len(x1_fin_train)}\t {results[0]:.4f}\t {[round(results[1][i],4) for i in range(0,nvar)]}\t \
              {results[2]:.4f}\t {results[3]:.4f} \t {results[4]:.4f} \n')   
    f1.close()   
    
    
    
    results=HEUR1(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MSE][0],measures_results[best_MSE][1],nvar)
    f2 = open(results2, "a")
    f2.write(f'{dataname}\t HEUR1\t {pesos[k]}\t {measures_results[best_MSE][0]}\t {measures_results[best_MSE][1]}\t {len(x1_fin_train)}\t {results[0]:.4f}\t {[round(results[1][i],4) for i in range(0,nvar)]}\t \
              {results[2]:.4f}\t {results[3]:.4f} \t {results[4]:.4f}   \n')   
    f2.close()     
   
  