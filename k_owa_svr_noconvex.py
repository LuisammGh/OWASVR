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
#parameters, weight, time limit
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


#heuristic for initial solution
# =============================================================================
def heuristic(TipoPeso,x,y,x_test,y_test,C,epsilon,sigma):
  #--------PESOS LAMBDA------------------------------------------------------#
    n=len(x)
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
    svr=SVR(kernel='rbf',gamma=1/(2*pow(sigma,2)), C=C,epsilon=epsilon)
    svr.fit(x,y)
    pred=svr.predict(x)
    #svr.fit(x.reshape(-1,1),y)
    #pred=svr.predict(x.reshape(-1,1))

    errores=[abs(pred[i]-y[i]) for i in range(0,n)]
    argerrores=np.argsort(errores)

    m = gp.Model("OWA_heuristic")
    m.setParam('TimeLimit', timelim)


   
    a_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="a")
    a2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="a2")
    b_var=m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name="b")
    xi_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi")
    xi2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi2")
    

    objective=1/2*gp.quicksum((a_vars[i]-a2_vars[i])*(a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for i in range(0,n) for j in range(0,n))+C*gp.quicksum(l[i]*(xi_vars[argerrores[i]]+xi2_vars[argerrores[i]]) for i in range(0,n))
    m.addConstrs(y[i]-gp.quicksum((a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for j in range(0,n))-b_var<=epsilon+xi_vars[i] for i in range(0,n))

    m.addConstrs(-y[i]+gp.quicksum((a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for j in range(0,n))+b_var<=epsilon+xi2_vars[i] for i in range(0,n))
    
    
    m.addConstr(gp.quicksum(a_vars[i]-a2_vars[i] for i in range(0,n))==0)
    m.addConstrs(a_vars[argerrores[k]]<=C*l[k] for k in range(0,n))
    m.addConstrs(a2_vars[argerrores[k]]<=C*l[k] for k in range(0,n))

    tsol1=time.time()
    m.params.NonConvex = 2
    m.params.NumericFocus=2
    m.setObjective(objective, GRB.MINIMIZE)
    m.write('model.lp')
    m.optimize()
    tsol2=time.time()
    status=m.status    
    INTERCEPT=b_var.X
    xi_sol=[xi_vars[i].X for i in range(0,n)]
    xi2_sol=[xi2_vars[i].X for i in range(0,n)]
    xi0_sol=[xi_vars[i].X+xi2_vars[i].X for i in range(0,n)]
    theta_sol=np.sort(xi0_sol)
    orden=np.argsort(xi0_sol)
    a_sol=[a_vars[i].X for i in range(0,n)]
    a2_sol=[a2_vars[i].X for i in range(0,n)]

    objvalue=m.ObjVal
    m.reset()
    nn=len(x_test)
    

    stime=tsol2-tsol1
    return(INTERCEPT,stime,status,objvalue,xi_sol,xi2_sol,xi0_sol,theta_sol,orden,a_sol,a2_sol)
#===============================================================================================================#


#NOMON======================================================================================================#
def owa_svr_nomon_kernel(TipoPeso,x,y,x_test,y_test,C,epsilon,sigma,inter,xi,xi2,theta,ordenz,a,a2):
  #--------PESOS LAMBDA------------------------------------------------------#
    n=len(x)
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
    m = gp.Model("OWA_SVR_NOMON_KERNEL")
    m.setParam('TimeLimit', timelim)
    
    a_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="a")
    a2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="a2")
    b_var=m.addVar(vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,name="b")
    theta_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="theta")
    z_vars=m.addVars(range(0,n),range(0,n),vtype=GRB.BINARY,name="z")
    xi_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi")
    xi2_vars=m.addVars(range(0,n),vtype=GRB.CONTINUOUS,lb=0,name="xi2")
    z_auxi=m.addVars(range(0,n),range(0,n),vtype=GRB.CONTINUOUS,lb=0,ub=1,name="z_auxi")

    objective=1/2*gp.quicksum((a_vars[i]-a2_vars[i])*(a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for i in range(0,n) for j in range(0,n))


    objective+=C*gp.quicksum(l[i]*theta_vars[i] for i in range(0,n))

    m.addConstrs(gp.quicksum(z_vars[i,j] for i in range(0,n))==1 for j in range(0,n))
    m.addConstrs(gp.quicksum(z_vars[i,j] for j in range(0,n))==1 for i in range(0,n))

    m.addConstrs(y[i]-gp.quicksum((a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2))) for j in range(0,n))-b_var<=epsilon+xi_vars[i] for i in range(0,n))

    m.addConstrs(-y[i]+gp.quicksum((a_vars[j]-a2_vars[j])*np.exp(-np.dot(x[i]-x[j],x[i]-x[j])/(2*pow(sigma,2)))for j in range(0,n))+b_var<=epsilon+xi2_vars[i] for i in range(0,n))
    
    m.addConstrs(a_vars[i]<=C*gp.quicksum(l[k]*z_vars[i,k] for k in range(0,n)) for i in range(0,n))
    m.addConstrs(a2_vars[i]<=C*gp.quicksum(l[k]*z_vars[i,k] for k in range(0,n)) for i in range(0,n))
    m.addConstr(gp.quicksum(a_vars[i]-a2_vars[i] for i in range(0,n))==0)
    
    m.addConstrs(z_auxi[i,k]==gp.quicksum(z_vars[i,j] for j in range(0,k)) for i in range(0,n) for k in range(0,n))

    for i in range(0,n):
        for k in range(0,n):
            m.addGenConstrIndicator(z_auxi[i,k],1,theta_vars[k]-xi_vars[i]-xi2_vars[i]>=0)
    


    #-sol ini---#
    b_var.Start=inter
    for i in range(0,n):
        xi_vars[i].Start=xi[i]
        xi2_vars[i].Start=xi2[i]
        theta_vars[i].Start=theta[i]
        a_vars[i].Start=a[i]
        a2_vars[i].Start=a2[i]
         
    for i in range(0,n):
        z_vars[ordenz[i],i].Start=1
        for j in range(0,n):
                if i!=j:
                    z_vars[ordenz[i],j].Start=0

    tsol1=time.time()
    m.params.NonConvex = 2
    m.setObjective(objective, GRB.MINIMIZE)
    #m.setParam('Threads',4)
    m.write('model.lp')
    m.optimize()
    tsol2=time.time()

    obj=m.ObjVal
    INTERCEPT=b_var.X 
    status=m.status  
    gap=m.MIPGAP
    nn=len(x_test)
    
    y_pred=[INTERCEPT+sum((a_vars[i].X-a2_vars[i].X)*np.exp(-np.dot(x[i]-x_test[j],x[i]-x_test[j])/(2*pow(sigma,2)))for i in range(0,n)) for j in range(0,nn)]
    m.reset()
    mae=mean_absolute_error(y_test, y_pred)
    mse=mean_squared_error(y_test,y_pred)

    stime=tsol2-tsol1
    return(obj,INTERCEPT,tsol2-tsol1,status,gap,mae,mse,y_test,y_pred)
#====================================================================================================================================#
   
results0="results_KOWASVR_Fold_"+dataname+".txt"
if(os.path.isfile(results0)==False):
    f0 = open(results0, "a")
    f0.write(f'Data \t Model\t Weight \t C \t eps\t sigma\t fold \t size \t obj \t intercept\t  \t time \t status \t GAP \t MAE \t MSE  \n')
    f0.close()

    
results1="Best_Mae.txt"
if(os.path.isfile(results1)==False):
    f1= open(results1, "a")
    f1.write(f' Data\t Model \t Weight \t C \t eps\t sigma \t size \t obj \t intercept \t time \t status \t GAP \t MAD \t MSE   \n')      
    f1.close()
    
results2="Best_Mse.txt"
if(os.path.isfile(results2)==False):
    f2= open(results2, "a")
    f2.write(f'Data\t Model \t Weight \t C \t eps \t sigma\t size \t obj \t intercept \t time \t status \t GAP  \t MAD \t MSE  \n')
    f2.close()

    
results3="y_test_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results3)==False):
    f3= open(results3, "a")
    f3.write(f'Data\t Model \t Weight\t C\t eps\t fold  \t y_test \n')                    
    f3.close()

results4="y_pred_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results4)==False):
    f4= open(results4, "a")
    f4.write(f'Data\t Model \t Weight\t C\t eps\t fold  \t y_pred \n')                    
    f4.close()
    
results5="y_error_SVR_Fold_"+dataname+".txt"
if(os.path.isfile(results5)==False):
    f5= open(results5, "a")
    f5.write(f'Data\t Model \t Weight\t C\t eps\t fold  \t error \n')                    
    f5.close()
            
for k in range(3,6):
    measures_results=[]
    for ii in range(0,len(C)):
        for j in range(0,len(eps)):
            for m in range(0,len(sigma)): #SIGMA
                times=[]
                MAE=[]
                MSE=[]
                for i in range(0,nfold):
                    print(C[ii]," ",eps[j]," ",i," ",pesos[k]," ",sigma[m])
                    timelim=900
                    result0=heuristic(k,x1_train[i],x2_train[i],x1_test[i],x2_test[i],C[ii],eps[j],sigma[m])              
                    results=owa_svr_nomon_kernel(k,x1_train[i],x2_train[i],x1_test[i],x2_test[i],C[ii],eps[j],sigma[m],result0[0],result0[4],result0[5],result0[7],result0[8],result0[9],result0[10])
                    f0 = open(results0, "a")
                    f0.write(f' {dataname}\t K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t {sigma[m]}\t{i}\t {len(all_train_fold[i])}\t {results[0]:.4f}\t {results[1]:.4f}\t \
                             {results[2]:.4f}\t {results[3]:.4f}\t {results[4]:.4f}\t {results[5]:.4f} \t {results[6]:.4f}  \n')   
                    f0.close()   
                    f3= open(results3, "a")
                    f3.write(f'{dataname}\t K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                    for ll in results[7]:
                        f3.write(f'\t {ll:.4f}')
                    f3.write(f'\n')
                    f3.close() 
                
                    f4= open(results4, "a")
                    f4.write(f'{dataname}\t K-OWA-SVR\t {pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                    for ll in results[8]:
                        f4.write(f'\t {ll:.4f}')
                    f4.write(f'\n')
                    f4.close()  
                
                
                    er=results[7]-results[8]
                
                    f5= open(results5, "a")
                    f5.write(f'{dataname}\t K-OWA-SVR\t{pesos[k]}\t {C[ii]}\t {eps[j]}\t{i}') 
                    for ll in er:
                        f5.write(f'\t {ll:.4f}')
                    f5.write(f'\n')
                    f5.close()  
                    times.append(results[2])
                    MAE.append(results[5])
                    MSE.append(results[6])

                measures_results.append([C[ii],eps[j],sigma[m],np.mean(times),np.mean(MAE),np.mean(MSE)])
    best_MAE=[np.argmin(i) for i in zip(*measures_results)][4]
    best_MSE=[np.argmin(i) for i in zip(*measures_results)][5]
    


    timelim=1800
    result0=heuristic(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MAE][0],measures_results[best_MAE][1],measures_results[best_MAE][2])              
    results=owa_svr_nomon_kernel(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,
    measures_results[best_MAE][0],measures_results[best_MAE][1],measures_results[best_MAE][2],result0[0],result0[4],result0[5],result0[7],result0[8],result0[9],result0[10])

    f1= open(results1, "a")
    f1.write(f'{dataname}\t K-OWA-SVR\t {pesos[k]}\t {measures_results[best_MAE][0]}\t {measures_results[best_MAE][1]}\t {measures_results[best_MAE][2]}\t{len(x1_fin_train)}\t {results[0]:.4f}\t {results[1]:.4f}\t \
              {results[2]:.4f}\t {results[3]:.4f} \t {results[4]:.4f} \t {results[5]:.4f} \t {results[6]:.4f}   \n')   
    f1.close()   
    
    
    
    timelim=1800
    result0=heuristic(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MSE][0],measures_results[best_MSE][1],measures_results[best_MSE][2])              
    results=owa_svr_nomon_kernel(k,x1_fin_train,x2_fin_train,x1_fin_test,x2_fin_test,measures_results[best_MSE][0],measures_results[best_MSE][1],measures_results[best_MSE][2],result0[0],result0[4],result0[5],result0[7],result0[8],result0[9],result0[10])
    f2 = open(results2, "a")
    f2.write(f'{dataname}\t K-OWA-SVR\t {pesos[k]}\t {measures_results[best_MSE][0]}\t {measures_results[best_MSE][1]}\t {measures_results[best_MSE][2]}\t {len(x1_fin_train)}\t {results[0]:.4f}\t {results[1]:.4f}\t \
              {results[2]:.4f}\t {results[3]:.4f} \t {results[4]:.4f} \t {results[5]:.4f} \t {results[6]:.4f} \n')   
    f2.close() 


 



