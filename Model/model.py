"""
Create a MILP forumlation for a repatriation scheduling problem based on the following paper:
    - https://www.sciencedirect.com/science/article/pii/S0957417422002019#d1e3348
"""

import gurobipy as gp
import numpy as np
from tqdm import tqdm


class RSPModel:
    def __init__(self, data, w, U=2, QC=2000):
        self.model = gp.Model()
        self.P = data
        self.M, self.N = self.P.shape
        self.w = w
        self.U = U
        self.C = np.empty((self.U,))
        self.QC = QC
        self.V = 10e6

        self.alpha = {}
        self.epsilon = {}
        self.L = {}
        self.P_cum = {}
        self.P_hat_cum = {}
        self.R = {}
        self.x = {}

    def __repr__(self):
        return f'RSPModel({self.P}, {self.w}, {self.U}, {self.QC})'

    def set_airplane_capacity(self):
        self.C.fill(300)

    def create_variables(self):
        for i in range(self.M):
            self.epsilon[i] = self.model.addVar(vtype=gp.GRB.BINARY, name="epsilon[%s]"%(i))

        for i in range(self.M):
            self.L[i] = self.model.addVar(vtype=gp.GRB.INTEGER, name="L[%s]" % (i))

        for i in range(self.M):
            for j in range(self.N):
                self.alpha[i, j] = self.model.addVar(vtype=gp.GRB.BINARY, name="alpha[%s,%s]"%(i,j))

        for i in range(self.M):
            for j in range(self.N):
                self.P_cum[i, j] = gp.quicksum(self.P[i, u] for u in range(j + 1))

        for i in range(self.M):
            for j in range(self.N):
                self.P_hat_cum[i, j] = self.model.addVar(vtype=gp.GRB.INTEGER, name="P_hat_cum[%s,%s]" % (i, j))

        for i in range(self.M):
            for j in range(self.N):
                self.R[i, j] = self.model.addVar(vtype=gp.GRB.INTEGER, name="R[%s,%s]" % (i, j))

        for i in range(self.M):
            for k in range(self.U):
                self.x[i, k] = self.model.addVar(vtype=gp.GRB.BINARY, name="xik[%s,%s]"%(i,k))

        self.model.update()

    def create_objective(self):
        self.obj = gp.LinExpr()
        self.obj += gp.quicksum(self.w[j] * self.R[i, j] for i in range(self.M) for j in range(self.N))
        self.model.setObjective(self.obj, gp.GRB.MAXIMIZE)
        self.model.update()

    def create_constraints(self):
        # (3)
        for i in range(self.M):
            LHS, RHS = gp.LinExpr(), gp.LinExpr()
            LHS += self.P_cum[i, self.N - 1] - gp.quicksum(self.C[k] * self.x[i, k] for k in range(self.U))
            RHS += self.V * (self.epsilon[i] - 1)
            self.model.addConstr(LHS >= RHS, name='(3)[%s]' % (i))

        # (4)
        for i in range(self.M):
            LHS, RHS = gp.LinExpr(), gp.LinExpr()
            LHS += self.P_cum[i, self.N - 1] - gp.quicksum(self.C[k] * self.x[i, k] for k in range(self.U))
            RHS += self.V * self.epsilon[i]
            self.model.addConstr(LHS <= RHS, name='(4)[%s]' % (i))

        # (5)
        for i in range(self.M):
            LHS, RHS = gp.LinExpr(), gp.LinExpr()
            LHS += self.L[i]
            RHS += gp.quicksum(self.C[k] * self.x[i, k] for k in range(self.U)) + self.V * (1 - self.epsilon[i])
            self.model.addConstr(LHS <= RHS, name='(5)[%s]' % (i))

        # (6)
        for i in range(self.M):
            LHS, RHS = gp.LinExpr(), gp.LinExpr()
            LHS += self.L[i]
            RHS += gp.quicksum(self.C[k] * self.x[i, k] for k in range(self.U)) + self.V * (self.epsilon[i] - 1)
            self.model.addConstr(LHS >= RHS, name='(6)[%s]' % (i))

        # (7)
        for i in range(self.M):
            LHS, RHS = gp.LinExpr(), gp.LinExpr()
            LHS += self.L[i]
            RHS += self.P_cum[i, self.N - 1] + self.V * self.epsilon[i]
            self.model.addConstr(LHS <= RHS, name='(7)[%s]' % (i))

        # (8)
        for i in range(self.M):
            LHS, RHS = gp.LinExpr(), gp.LinExpr()
            LHS += self.L[i]
            RHS += self.P_cum[i, self.N - 1] - self.V * self.epsilon[i]
            self.model.addConstr(LHS >= RHS, name='(8)[%s]' % (i))

        # (9)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.L[i] - self.P_cum[i, j]
                RHS += self.V * (self.alpha[i, j] - 1)
                self.model.addConstr(LHS >= RHS, name='(9)[%s,%s]' % (i, j))  # should be strict inequality

        # (10)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.L[i] - self.P_cum[i, j]
                RHS += self.V * self.alpha[i, j]
                self.model.addConstr(LHS <= RHS, name='(10)[%s,%s]' % (i, j))

        # (11)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.P_hat_cum[i, j]
                RHS += (1 - self.alpha[i, j]) * self.V
                self.model.addConstr(LHS <= RHS, name='(11)[%s,%s]' % (i, j))

        # (12)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.P_hat_cum[i, j]
                RHS += (self.alpha[i, j] - 1) * self.V
                self.model.addConstr(LHS >= RHS, name='(12)[%s,%s]' % (i, j))

        # (13)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.P_hat_cum[i, j]
                RHS += self.P_cum[i, j] - self.L[i] + self.alpha[i, j] * self.V
                self.model.addConstr(LHS <= RHS, name='(13)[%s,%s]' % (i, j))

        # (14)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.P_hat_cum[i, j]
                RHS += self.P_cum[i, j] - self.L[i] - self.alpha[i, j] * self.V
                self.model.addConstr(LHS >= RHS, name='(14)[%s,%s]' % (i, j))

        # (15)
        for i in range(self.M):
            for j in range(self.N):
                LHS, RHS = gp.LinExpr(), gp.LinExpr()
                LHS += self.R[i, j]
                if j != 0:
                    RHS += self.P_cum[i, j] - self.P_hat_cum[i, j] - self.P_cum[i, j - 1] + self.P_hat_cum[i, j - 1]
                elif j == 0:
                    RHS += self.P_cum[i, j] - self.P_hat_cum[i, j]
                self.model.addConstr(LHS == RHS, name='(15)[%s,%s]' % (i, j))

        # (16)
        self.model.addConstr(gp.quicksum(self.L[i] for i in range(self.M)) <= self.QC, name='(16)')

        # (17)
        for k in range(self.U):
            self.model.addConstr(gp.quicksum(self.x[i, k] for i in range(self.M)) <= 1, name='(17)[%s]' % (k))

        self.model.update()

    def solve(self):
        self.model.Params.LogToConsole = 0

        self.set_airplane_capacity()
        self.create_variables()
        self.create_objective()
        self.create_constraints()

        self.model.update()
        self.model.optimize()
        if self.model.status == gp.GRB.INF_OR_UNBD or self.model.status == gp.GRB.INFEASIBLE:
            self.model.computeIIS()
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print('%s' % c.constrName)
        else:
            self.model.write(f'results/model_{self.U}_{self.QC}.lp')


if __name__ == '__main__':
    us = [2,4,6,8,8,10,10,12,12,12,14]
    QCs = [2000,2000,2000,2000,2500,2500,3000,3000,3500,4000,4000]
    df = np.genfromtxt('test_data.csv', delimiter=';', skip_header=1)

    results = []

    for i in tqdm(range(len(us))):
        RSP = RSPModel(data=df[:5,:], w=df[5,:], U=us[i], QC=QCs[i])
        RSP.solve()
        results.append([RSP.U, RSP.QC, RSP.model.objVal])

    print(results)
