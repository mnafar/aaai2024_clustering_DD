# aaai2024_clustering_DD
 

The associated code is an implementation of the paper: 
"Using Clustering to Strengthen Decision Diagram Bounds for Discrete Optimization".

This code is designed to compile BDDs for 5 different problems;
1. 0/1 Knapsack, 
2. Multidimensional Knapsack, 
3. Weighted Number of Tardy Jobs on a Single Machine,
4. Sum of Cubed Job Completion Times on Two Identical Machines,
5. Total Weighted Job Completion Time on Two Identical Machines.

File "Clustering-based node selection in DD.jl" is a Pluto Notebook containing 4 cells; 

first cell, which is a very highly generic written code, is the related code 
that implements all the needed functions and structures,

second cell is the running code for building approximate BDDs for problem 1,

third cell is the running code for building approximate BDDs for problrm 2,

forth cell is the running code for building approximate BDDs for problems 3, 4, and 5. 


To install Julia please visit the following link:
https://julialang.org/ .

For an introduction to Pluto Notebook and start using it visit the following link:
https://plutojl.org/ .

Note on running the code:

To reproduce the results that are reported in the paper you just need to run
the second, third, and forth cells in the code. There are not any complicated 
instructions for running the code and the code is self contained. The only thing that 
you need to pay attention to when running the code is the addresses of the files, which
you must adjust according to the location of the instance folders on your device. 

Notes about the instances:

There are 3 folders containing the instances for problems;

1. KP_instances, 2. MKP_instances, 3. Scheduling_instances. 
Every folder contains the instance files where the optimum solution of 
every instance is also reported in the instance file.
