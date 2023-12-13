import numpy as np
from tqdm import tqdm
import pickle

class DataGenerator:
    def __init__(self, m=None, n=None, u=None, fq=None, from_array=None):
        if from_array is not None:
            self.M = from_array[0]
            self.N = from_array[1]
            self.U = from_array[2]
            self.QC = from_array[3]
            self.C = from_array[4]
            self.P = from_array[5]
            self.w = from_array[6]
            self.fq = 1
        else:
            # np.random.seed(0)
            self.M = m
            self.N = n
            self.U = u
            self.fq = fq

            self.C = np.empty((self.U,))
            capacity = [200, 250, 300, 350]
            for i in range(len(self.C)):
                self.C[i] = np.random.choice(capacity)
            self.QC = self.fq * np.sum(self.C)

            self.P = np.empty((self.M, self.N))
            for i in range(self.M):
                for j in range(self.N):
                    self.P[i, j] = int(np.random.uniform(0.5, 1.5) * np.average(self.C))

            self.w = list(range(1, self.N + 1))[::-1]

    def __repr__(self):
        return f'{self.M}-{self.U}-{self.N}-{self.fq}'

    def store(self):
        save_arr = [self.M, self.N, self.U, self.QC, self.C, self.P, self.w]
        pickle.dump(save_arr, open(f'data/{self}.pkl', 'wb'))

    @staticmethod
    def recover(data) -> 'DataGenerator':
        return DataGenerator(from_array=data)


    @staticmethod
    def from_file(filename, u, qc, capacity): # -> 'DataGenerator':
        df = np.genfromtxt(filename, delimiter=';', skip_header=1)
        C = np.empty((u, ))
        C.fill(capacity)
        arr = [df[:-1, :].shape[0], df[:-1, :].shape[1], u, qc, C, df[:-1, :], df[-1, :]]
        return DataGenerator(from_array=arr)



if __name__ == '__main__':
    cities = [150, 200, 250]
    airplanes = [600, 800, 1000]
    groups = [3, 6, 9]
    omega = [0.8, 1.2]
    with tqdm(total=len(cities) * len(airplanes) * len(groups) * len(omega)) as pbar:
        for c in cities:
            for a in airplanes:
                for g in groups:
                    for o in omega:
                        data = DataGenerator(c, g, a, o)
                        data.store()
                        pbar.update(1)

    with open('data/150-600-3-0.8.pkl', 'rb') as f:
        loaded_array = pickle.load(f)
    DATA = DataGenerator.recover(loaded_array)
    print(DATA.M, DATA.N, DATA.U, DATA.QC, DATA.C, DATA.P, DATA.w)
