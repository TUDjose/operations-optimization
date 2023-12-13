"""
Create a MILP forumlation for a repatriation scheduling problem based on the following paper:
    - https://www.sciencedirect.com/science/article/pii/S0957417422002019#d1e3348
"""
import os
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Model.data_generator import DataGenerator
import pickle


class RSPModel:
    def __init__(self, data: DataGenerator):
        self.model = gp.Model()
        self.P = data.P
        self.M = data.M
        self.N = data.N
        self.w = data.w
        self.U = data.U
        self.C = data.C
        self.QC = data.QC
        self.V = 1e6

        self.alpha = {}
        self.epsilon = {}
        self.L = {}
        self.P_cum = {}
        self.P_hat_cum = {}
        self.R = {}
        self.x = {}

    def __repr__(self):
        return f'RSPModel({self.P}, {self.w}, {self.U}, {self.QC})'

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

    def solve(self, print_log=False):
        self.model.Params.LogToConsole = 0 if not print_log else 1

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
            self.model.write(f'Model/results/model_{self.U}_{self.QC}.lp')


if __name__ == '__main__':
    results = []

    dg = DataGenerator.from_file('../tests/test_data.csv', 14, 4000, 300)
    RSP = RSPModel(dg)
    RSP.solve()
    print(RSP.model.objVal)

    with open(f'data/150-600-3-0.8.pkl', 'rb') as f:
        loaded_array = pickle.load(f)
    dg = DataGenerator.recover(loaded_array)
    RSP = RSPModel(dg)
    RSP.solve()
    print(RSP.model.objVal)
    print(sum(RSP.C))

    # # this will take forever
    # with tqdm(total=len(os.listdir('data'))) as pbar:
    #     for filename in os.listdir('data'):
    #         print(filename)
    #         with open(f'data/{filename}', 'rb') as f:
    #             loaded_array = pickle.load(f)
    #         RSP = RSPModel(DataGenerator.recover(loaded_array))
    #         RSP.solve()
    #         results.append([RSP.U, RSP.QC, RSP.model.objVal])
    #         pbar.update(1)
    #
    #
    # results = np.array(results)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_trisurf(results[:, 0], results[:, 1], results[:, 2], cmap='viridis', linewidth=0.2, edgecolor='k')
    # plt.show()
