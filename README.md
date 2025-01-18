# OWASVR
Exact and heuristic approaches for OWA-SVR
This repository contains the following codes:
1.	HEUR1.py: Heuristic approach for linear OWA-SVR in a fold cross validation. The description of this heuristic can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
2.	HEUR2.py: Heuristic approach for linear OWA-SVR in a fold cross validation. The description of this heuristic can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
3.	HEURK.py: Heuristic approach for OWA-SVR using the Gaussian Kernel in a fold cross validation. The description of this heuristic can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
4.	owa_svr_convex.py: Model for linear OWA-SVR using convex weights in a fold cross validation. The description of this model can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
5.	owa_svr_noconvex.py: Model for linear OWA-SVR using no convex weights in a fold cross validation. The description of this model can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
6.	k_owa_svr_convex.py: Model for OWA-SVR using convex weights and Gaussian Kernel in a fold cross validation. The description of this model can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
7.	k_owa_svr_noconvex.py: Model for OWA-SVR using no convex weights and Gaussian Kernel in a fold cross validation. The description of this model can be found in: Ordered Weighted Average Support Vector Regression, L.I. Martínez-Merino, J. Puerto, A.M. Rodríguez-Chía [REF]
These codes are implemented in Python and solved using Gurobi. The user provides the following data:
•	Name of the data file. Example: data/sonar_scale.txt
•	Number of folds for the fold cross validation. Example:5
•	Number of considered variables. Example: 1
The used parameter values are the following:
•	C= 0.01,0.1,1,10,100
•	epsilon= 0,0.01,0.1,1,10
•	sigma= 0.01,0.1,1,10,100
•	OWA operators: sum of all deviations, sum of the k largest deviations, largest deviation, sum of the k smallest deviations, median deviation and sum of the central deviation values.
Besides, folder "data" include some examples of datasets for regression.
