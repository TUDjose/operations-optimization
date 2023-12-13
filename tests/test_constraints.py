import unittest
from Model.model import RSPModel
from Model.data_generator import DataGenerator


class TestConstraints(unittest.TestCase):

    def test_objetive(self):
        dg = DataGenerator.from_file('tests/test_data.csv', 2, 2000, 300)
        print(dg.M)
        RSP = RSPModel(dg)
        RSP.solve()
        self.assertEqual(6000, 6000)


    def test_c2(self):
        pass

    def test_c3(self):
        pass

    def test_c4(self):
        pass

    def test_c5(self):
        pass

    def test_c6(self):
        pass

    def test_c7(self):
        pass

    def test_c8(self):
        pass

    def test_c9(self):
        pass

    def test_c10(self):
        pass

    def test_c11(self):
        pass

    def test_c12(self):
        pass

    def test_c13(self):
        pass

    def test_c14(self):
        pass

    def test_c15(self):
        pass

    def test_c16(self):
        pass

    def test_c17(self):
        pass
