"""
Create a MILP forumlation for a repatriation scheduling problem based on the following paper:
    - https://www.sciencedirect.com/science/article/pii/S0957417422002019#d1e3348
"""

import gurobipy as gp
import numpy as np
import csv
import pandas as pd

# read data from csv file
df = np.genfromtxt('test_data.csv', delimiter=';', skip_header=1)

# Create a new model
model = gp.Model()

# SETS
M = 5     # set of cities
N = 4     # set of priority groups
U = 2    # set of airplanes

# PROBLEM INPUTS
C = np.empty((U, ))     # data structure for capacity of airplane in k ∈ U
P = np.empty((M, N))     # data structure for number of individuals of group j ∈ N in city i ∈ M
QC = 2000    # number of available quarantine locations
w = np.empty((N, ))      # importance of citizens in group j ∈ N

def set_airplane_capacity():
    C.fill(300)

def set_individuals():
    for i in range(M):
        for j in range(N):
            P[i][j] = df[i][j]

def set_importance():
    for j in range(N):
        w[j] = df[-1][j]

set_airplane_capacity()
set_individuals()
set_importance()

# MILP VARIABLES

alpha = {}
for i in range(M):
    for j in range(N):
        alpha[i, j] = model.addVar(vtype=gp.GRB.BINARY, name="alpha[%s,%s]"%(i,j))

epsilon = {}
for i in range(M):
    epsilon[i] = model.addVar(vtype=gp.GRB.BINARY, name="epsilon[%s]"%(i))

L = {}
for i in range(M):
    L[i] = model.addVar(vtype=gp.GRB.INTEGER, name="L[%s]"%(i))

P_cum = {}
for i in range(M):
    for j in range(N):
        P_cum[i, j] = model.addVar(vtype=gp.GRB.INTEGER, name="P_cum[%s,%s]"%(i,j))

P_hat_cum = {}
for i in range(M):
    for j in range(N):
        P_hat_cum[i, j] = model.addVar(vtype=gp.GRB.INTEGER, name="P_hat_cum[%s,%s]"%(i,j))

R = {}
for i in range(M):
    for j in range(N):
        R[i, j] = model.addVar(vtype=gp.GRB.INTEGER, name="R[%s,%s]"%(i,j))

xik = {}
for i in range(M):
    for k in range(U):
        xik[i, k] = model.addVar(vtype=gp.GRB.BINARY, name="xik[%s,%s]"%(i,k))

model.update()



# OBJECTIVE FUNCTION
obj = gp.LinExpr()
obj += gp.quicksum([w[j] * R[i, j] for i in range(M) for j in range(N)])
model.setObjective(obj, gp.GRB.MAXIMIZE)
model.update()
model.optimize()
