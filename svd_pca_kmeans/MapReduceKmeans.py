from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
import simplejson

class MRKmeans(MRJob):
    #OUTPUT_PROTOCOL: object = mrjob.protocol.RawProtocol

    def configure_args(self):
        # Define input file, output file and number of iteration
        super(MRKmeans, self).configure_args()
        self.add_file_arg('--infile')
        self.add_file_arg('--outfile')
        self.add_passthru_arg('-n', '--iterations', default=10, type=int, help='number of iterations')


    def get_centroids(self):
        Centroid = np.loadtxt('/Users/ruhulislam/PycharmProjects/pythonProject/Centroid.txt', delimiter=',')
        #Centroid = np.array([0.85354,1.16891,2.75221,3.86218,4.88659])
        #Centroid = np.array([0.85354, 1.16891, 2.75221])
        #Centroid = 0.85354,2.16891,2.75221,3.86218,4.88659,6.01489,6.74455,8.00533,8.88937
        return Centroid

        # Centroid = np.loadtxt(self.options.infile, delimiter=',')
        # return Centroid

    def write_centroids(self, Centroid):
        #np.savetxt(self.options.outfile, Centroid[None], fmt='%.5f', delimiter=',')
        #X = np.arange(100)
        X = np.array(Centroid)
        np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/KMCentroid.txt', X, fmt='%.5f', delimiter=',')


    def mapper_readataa(self, _, line):
        '''
        Mapper function:
        Here I am loading the coordinates and
        initial centroids to calculate teh distance
        from each centroids and retagging the clusters
        to the respective data points. Also passing the Cluster Id
        as key and the data id and coordinates as values to the
        combiner funciton.
        '''
        data_ID, Cluster_ID, Coordinate = line.split('|')
        Coordinate = Coordinate.strip('\r\n')
        Coordinate_array = np.array(Coordinate.split(','), dtype=float)
        global Centroid
        Centroid = self.get_centroids()
        Centroid_arr = np.reshape(Centroid, (-1, len(Coordinate_array)))
        global nclass
        nclass = Centroid_arr.shape[0]
        global ndim
        ndim = Centroid_arr.shape[1]
        Distance = ((Centroid_arr - Coordinate_array) ** 2).sum(axis=1)
        Cluster_ID = str(Distance.argmin() + 1)
        Coord_arr = Coordinate_array.tolist()
        yield Cluster_ID, (data_ID, Coord_arr)

    def combiner_nodecal(self, Cluster_ID, values):
        '''
        Combiner function:
        Here in the combiner function I am combining the keys from each
        machine that is come from the mapper and giving output the
        Cluster ID as key, members ID of that cluster, combination of
        samples with same key, member coordinates of that cluster as value.
        '''

        member = []
        Coordinate_set = []
        Coordinate_sum = np.zeros(ndim)
        for data_ID, Coordinate_array in values:
            Coordinate_set.append(','.join(str(e) for e in Coordinate_array))
            Coordinate_array = np.array(Coordinate_array, dtype=float)
            member.append(data_ID)
            Coordinate_sum += Coordinate_array
            Coordinate_sum = Coordinate_sum.tolist()
        yield Cluster_ID, (member, Coordinate_sum, Coordinate_set)

    def update_centroid(self, Cluster_ID, values):

        '''
        In this reducer method I am updaitng the centroids based on the
        calculated distance and making the output ready for
        alogn with taggin the ID as and cluster label.
        '''
        final_member = []
        final_Coord_set = []
        final_Coord_sum = np.zeros(ndim)
        clustered_data = []
        # f = open('/Users/ruhulislam/PycharmProjects/pythonProject/KMclusterd_output.txt', 'w')
        f = open('/Users/ruhulislam/PycharmProjects/pythonProject/KMclusterd_output.txt', 'a')
        # f1 = open('/Users/ruhulislam/PycharmProjects/pythonProject/KMclusterd_output1.txt', 'a')

        for member, Coord_sum, Coord_set in values:
            final_Coord_set += Coord_set
            Coord_sum = np.array(Coord_sum, dtype=float)
            final_member += member
            final_Coord_sum += Coord_sum

        n = len(final_member)
        new_Centroid = final_Coord_sum / n
        Centroid[ndim * (int(Cluster_ID) - 1): ndim * int(Cluster_ID)] = new_Centroid
        if int(Cluster_ID) == nclass:
            self.write_centroids(Centroid)

        for ID in final_member:
            ind = final_member.index(ID)
            # clustered_data.append(Cluster_ID)
            # clustered_data.append(final_Coord_set[ind])
            clustered_data1 = ID + ',' + Cluster_ID + ',' + final_Coord_set[ind] + ',' + "\n"
            # simplejson.dump(clustered_data1, f)
            #if self.options.iterations == 10:
            # f = open('/Users/ruhulislam/PycharmProjects/pythonProject/KMclusterd_output.txt', 'w')
            f.write(clustered_data1)
            yield None, (ID + '|' + Cluster_ID + '|' + final_Coord_set[ind])
            #yield ID, ('|' + Cluster_ID + '|' + final_Coord_set[ind])
        #np.savetxt('/Users/ruhulislam/PycharmProjects/pythonProject/outpcaBuitinRatio.txt', clustered_data)
        #simplejson.dump(clustered_data, f)
        # clustered_data2 = ID + '|' + Cluster_ID + '|' + final_Coord_set[ind] + ',' + "\n"
        # f1.write(clustered_data2)
        f.close()
        # f1.close()

    def steps(self):
        '''
        Here I ahve wrote the steps that will be run iteratively.
        The output from the reducer can be pused through the
        mapper as many times as I want to ensure optimum clustering of the
        data
        '''
        return [MRStep(mapper=self.mapper_readataa,
                       combiner=self.combiner_nodecal,
                       reducer=self.update_centroid)] * self.options.iterations


if __name__ == '__main__':
    MRKmeans.run()

"""
python MapReduceKmeans.py -r inline KMTestData_by_outpcaZWithIDCLID.txt
python MapReduceKmeans.py -r inline KMTestData_by_testdata.txt 

"""