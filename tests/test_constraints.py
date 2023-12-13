import unittest
import numpy as np
import numpy.testing as npt
from Model.model import RSPModel
from Model.data_generator import DataGenerator


class TestConstraints(unittest.TestCase):

    def test_objetive(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()
        self.assertEqual(RSP.model.objVal, 6000)

    def test_c2(self):
        P = np.array([[100, 200],
                      [200, 300]])
        dg = DataGenerator.from_array([2, 2, 1, 2000, [300], P, [2, 1]])
        RSP = RSPModel(dg)
        RSP.solve()
        npt.assert_array_equal(list(x.getValue() for x in RSP.P_cum.values()), [100, 300, 200, 500])

    def test_c3(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            LHS = RSP.P_cum[i, RSP.N-1] - sum(RSP.C[k] * RSP.x[i, k].x for k in range(RSP.U))
            RHS = 1e6 * (RSP.epsilon[i].x - 1)
            self.assertGreaterEqual(LHS.getValue(), RHS)

    def test_c4(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            LHS = RSP.P_cum[i, RSP.N-1] - sum(RSP.C[k] * RSP.x[i, k].x for k in range(RSP.U))
            RHS = 1e6 * RSP.epsilon[i].x
            self.assertLessEqual(LHS.getValue(), RHS)

    def test_c5(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            LHS = RSP.L[i] - sum(RSP.C[k] * RSP.x[i, k].x for k in range(RSP.U))
            RHS = 1e6 * (1- RSP.epsilon[i].x)
            self.assertLessEqual(LHS.getValue(), RHS)

    def test_c6(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            LHS = RSP.L[i] - sum(RSP.C[k] * RSP.x[i, k].x for k in range(RSP.U))
            RHS = 1e6 * (RSP.epsilon[i].x - 1)
            self.assertGreaterEqual(LHS.getValue(), RHS)

    def test_c7(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            LHS = RSP.L[i] - RSP.P_cum[i, RSP.N-1]
            RHS = 1e6 * RSP.epsilon[i].x
            self.assertLessEqual(LHS.getValue(), RHS)

    def test_c8(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            LHS = RSP.L[i] - RSP.P_cum[i, RSP.N - 1]
            RHS = -1e6 * RSP.epsilon[i].x
            self.assertGreaterEqual(LHS.getValue(), RHS)

    def test_c9(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                LHS = RSP.L[i] - RSP.P_cum[i, j]
                RHS = (RSP.alpha[i, j].x - 1) * RSP.V
                self.assertGreater(LHS.getValue(), RHS)

    def test_c10(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                LHS = RSP.L[i] - RSP.P_cum[i, j]
                RHS = RSP.alpha[i, j].x * RSP.V
                self.assertLessEqual(LHS.getValue(), RHS)

    def test_c11(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                LHS = RSP.P_hat_cum[i, j].x
                RHS = (1 - RSP.alpha[i, j].x) * RSP.V
                self.assertLessEqual(LHS, RHS)

    def test_c12(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                LHS = RSP.P_hat_cum[i, j].x
                RHS = (RSP.alpha[i, j].x - 1) * RSP.V
                self.assertGreaterEqual(LHS, RHS)

    def test_c13(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                LHS = RSP.P_hat_cum[i, j].x - RSP.P_cum[i, j] + RSP.L[i]
                RHS = RSP.alpha[i, j].x * RSP.V
                self.assertLessEqual(LHS.getValue(), RHS)

    def test_c14(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                LHS = RSP.P_hat_cum[i, j].x - RSP.P_cum[i, j] + RSP.L[i]
                RHS = - RSP.alpha[i, j].x * RSP.V
                self.assertGreaterEqual(LHS.getValue(), RHS)

    def test_c15(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for i in range(RSP.M):
            for j in range(RSP.N):
                if j == 0:
                    LHS = RSP.R[i, j].x
                    RHS = RSP.P_cum[i, j] - RSP.P_hat_cum[i, j].x
                else:
                    LHS = RSP.R[i, j].x
                    RHS = RSP.P_cum[i, j] - RSP.P_hat_cum[i, j].x - RSP.P_cum[i, j - 1] + RSP.P_hat_cum[i, j - 1].x
                self.assertEqual(LHS, RHS.getValue())

    def test_c16(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        LHS = sum(RSP.L[i].x for i in range(RSP.M))
        self.assertLessEqual(LHS, RSP.QC)

    def test_c17(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        RSP = RSPModel(dg)
        RSP.solve()

        for k in range(RSP.U):
            self.assertLessEqual(sum(RSP.x[i, k].x for i in range(RSP.M)), 1)
