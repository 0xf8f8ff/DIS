from mrjob.job import MRJob
from mrjob.step import MRStep
from sklearn.decomposition import PCA
import numpy as np


class MapReduceBiultinPCA(MRJob):

    def mapper_builtinPCA(self, _, line):
        data_ID, features = line.split('|')
        feature_data = features.strip('\r\n')
        Features_arr = np.array(feature_data.split(','), dtype=float)
        row = Features_arr.tolist()

        yield None, (data_ID, row)

    def reducer_builtinPCA(self, data_ID, values):
        dt = []
        for data_id, row in values:
            dt.append(row)

        val = []
        x = np.array(dt).T
        k = 2
        pca = PCA(n_components=k)
        z_pca = pca.fit_transform(x)
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/KMTestData_by_outpcaZWithIDCLID.txt', z_pca)
        X_approx_pca = pca.inverse_transform(z_pca)
        ratio_pca = np.mean((x - X_approx_pca).T.dot(x - X_approx_pca)) / np.mean(x.T.dot(x))
        val.append(ratio_pca)
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaBuitinRatio.txt', val)

    def steps(self):
        return [
            MRStep(mapper=self.mapper_builtinPCA),
            MRStep(reducer=self.reducer_builtinPCA)
        ]


if __name__ == '__main__':
    MapReduceBiultinPCA.run()

"""
python MapReduceBiultinPCA.py -r inline pcaorg_formatted.txt


"""