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
model = gp.Model("VRP")

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

x = {}
for i in range(M):
    for k in range(U):
        x[i, k] = model.addVar(vtype=gp.GRB.BINARY, name="xik[%s,%s]"%(i,k))

model.update()


# OBJECTIVE FUNCTION
# (1)
obj = gp.LinExpr()
obj += gp.quicksum(w[j] * R[i, j] for i in range(M) for j in range(N))
model.setObjective(obj, gp.GRB.MAXIMIZE)
model.update()

# CONSTRAINTS

V = 10e6    # big M

# (2)
for i in range(M):
    for j in range(N):
        model.addConstr(P_cum[i, j] == gp.quicksum(P[i, u] for u in range(j+1)), name='(2)[%s,%s]'%(i,j))

# (3)
for i in range(M):
    LHS, RHS = gp.LinExpr(), gp.LinExpr()
    LHS += P_cum[i, N-1] - gp.quicksum(C[k] * x[i, k] for k in range(U))
    RHS += V * (epsilon[i] - 1)
    model.addConstr(LHS >= RHS, name='(3)[%s]'%(i))

# (4)
for i in range(M):
    LHS, RHS = gp.LinExpr(), gp.LinExpr()
    LHS += P_cum[i, N-1] - gp.quicksum(C[k] * x[i, k] for k in range(U))
    RHS += V * epsilon[i]
    model.addConstr(LHS <= RHS, name='(4)[%s]'%(i))

# (5)
for i in range(M):
    LHS, RHS = gp.LinExpr(), gp.LinExpr()
    LHS += L[i]
    RHS += gp.quicksum(C[k] * x[i, k] for k in range(U)) + V * (1 - epsilon[i])
    model.addConstr(LHS <= RHS, name='(5)[%s]'%(i))

# (6)
for i in range(M):
    LHS, RHS = gp.LinExpr(), gp.LinExpr()
    LHS += L[i]
    RHS += gp.quicksum(C[k] * x[i, k] for k in range(U)) + V * (epsilon[i] - 1)
    model.addConstr(LHS >= RHS, name='(6)[%s]'%(i))

# (7)
for i in range(M):
    LHS, RHS = gp.LinExpr(), gp.LinExpr()
    LHS += L[i]
    RHS += P_cum[i, N-1] + V * epsilon[i]
    model.addConstr(LHS <= RHS, name='(7)[%s]'%(i))

# (8)
for i in range(M):
    LHS, RHS = gp.LinExpr(), gp.LinExpr()
    LHS += L[i]
    RHS += P_cum[i, N-1] - V * epsilon[i]
    model.addConstr(LHS >= RHS, name='(8)[%s]'%(i))

# (9)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += L[i] - P_cum[i, j]
        RHS += V * (alpha[i, j] - 1)
        model.addConstr(LHS >= RHS, name='(9)[%s,%s]'%(i,j))    # should be strict inequality

# (10)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += L[i] - P_cum[i, j]
        RHS += V * alpha[i, j]
        model.addConstr(LHS <= RHS, name='(10)[%s,%s]'%(i,j))

# (11)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += P_hat_cum[i, j]
        RHS += (1 - alpha[i, j]) * V
        model.addConstr(LHS <= RHS, name='(11)[%s,%s]'%(i,j))

# (12)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += P_hat_cum[i, j]
        RHS += (alpha[i, j] - 1) * V
        model.addConstr(LHS >= RHS, name='(12)[%s,%s]'%(i,j))

# (13)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += P_hat_cum[i, j]
        RHS += P_cum[i, j] - L[i] + alpha[i, j] * V
        model.addConstr(LHS <= RHS, name='(13)[%s,%s]'%(i,j))

# (14)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += P_hat_cum[i, j]
        RHS += P_cum[i, j] - L[i] - alpha[i, j] * V
        model.addConstr(LHS >= RHS, name='(14)[%s,%s]'%(i,j))

# (15)
for i in range(M):
    for j in range(N):
        LHS, RHS = gp.LinExpr(), gp.LinExpr()
        LHS += R[i, j]
        if j != 0:
            RHS += P_cum[i, j] - P_hat_cum[i, j] - P_cum[i, j-1] - P_hat_cum[i, j-1]
        elif j == 0:
            RHS += P_cum[i, j] - P_hat_cum[i, j]
        model.addConstr(LHS == RHS, name='(15)[%s,%s]'%(i,j))

# (16)
model.addConstr(gp.quicksum(L[i] for i in range(M)) <= QC, name='(16)')

# (17)
for k in range(U):
    model.addConstr(gp.quicksum(x[i, k] for i in range(M)) <= 1, name='(17)[%s]'%(k))

# (18)
# for i in range(M):
#     for k in range(U):
#         val = 0   # not sure how to get this value!!!
#         model.addConstr(x[i, k] == val, name='(18)[%s,%s]'%(i,k))

model.update()
model.optimize()
model.write('model.lp')

if model.status == gp.GRB.INF_OR_UNBD or model.status == gp.GRB.INFEASIBLE:
    model.computeIIS()
    model.write("infeasible.lp")
    print('\nThe following constraint(s) cannot be satisfied:')
    for c in model.getConstrs():
        if c.IISConstr:
            print('%s' % c.constrName)

