from mrjob.job import MRJob
from mrjob.step import MRStep
from sklearn.decomposition import PCA
import numpy as np
import sys


class MapReducePCAbySVD(MRJob):

    def mapper_init(self):
    #def __init__(self):
        self.message_id = ''
        self.in_body = False
        self.data_list = []

    def get_SVD(self, cov_matrix):
        u, s, v = np.linalg.svd(cov_matrix, full_matrices=False)
        return u, s, v

    def get_KComponents(self, u, x, k=2):
        z = np.dot(x, u[:, :k])
        u_reduced = u[:, :k]
        return z#, u_reduced

    def find_BestK(self, initial, step, u, x):
        for k in range(initial, 30, step):
            z = self.get_KComponents(u, x, k)
            U_reduced = u[:, :k]
            ratio = self.get_VarianceRatio(z, U_reduced, x, k)
            if ratio <= 0.0019:
                break
        return k, ratio

    def get_VarianceRatio(self, z, u, x, k):
        u = u[:, :k]
        X_approx_pca = np.dot(z, np.transpose(u))
        ratio = np.mean((x - X_approx_pca).T.dot(x - X_approx_pca)) / np.mean(x.T.dot(x))
        return ratio

    def mapper_PCAbySVD(self, _, line):
        data_ID, features = line.split('|')
        feature_data = features.strip('\r\n')
        Features_arr = np.array(feature_data.split(','), dtype=float)
        row = Features_arr.tolist()
        #self.data_list.append(row)

        yield None, (data_ID, row)

    def reducer_PCAbySVD(self, data_ID, values):

        dt = []
        for data_id, row in values:
            dt.append(row)

        val = []
        # x = self.data_list.to_numpy()
        x = np.array(dt)
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaX.txt', x)
        #x = x.T
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaXT.txt', x)
        cov_matrix = np.cov(x.T, bias=False)
        # cov_matrix = np.cov(x, bias=False)
        u, s, v = self.get_SVD(cov_matrix)
        k = 2
        z = self.get_KComponents(u, x, k)
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaZ.txt', z)

        clusterss = np.random.randint(1, 4, z.shape[0])
        ids = list(range(1, 1 + len(z)))
        ids = np.array(ids)  # .to_numpy()
        ids = ids.tolist()
        lowerdim_data_z_with_ID_CID = np.vstack((clusterss, z.T)).T
        lowerdim_data_z_with_ID_CID = np.vstack((ids, lowerdim_data_z_with_ID_CID.T)).T
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaZWithIDCLID.txt', lowerdim_data_z_with_ID_CID)

        ratio = self.get_VarianceRatio(z, u, x, k)
        best_k, best_ratio = self.find_BestK(1, 1, u, x)
        val.append(ratio)
        val.append(best_k)
        val.append(best_ratio)
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaRatio.txt', val)

    def steps(self):
        return [
            MRStep(mapper=self.mapper_PCAbySVD),
            MRStep(reducer=self.reducer_PCAbySVD)
        ]


if __name__ == '__main__':
    MapReducePCAbySVD.run()

"""
python MapReducePCAbySVD.py -r inline pcaorg_formatted.txt

"""