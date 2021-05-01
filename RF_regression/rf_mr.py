from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
import math
import numpy as np
import pandas as pd
import datetime
import sys

class RFTree:
    def __init__(self, training, testing, depth, min_samples, sample_features):
        self.training_set = np.array(training)
        self.testing_set = np.array(testing)
        self.tree_depth = depth
        self.min_samples = min_samples
        self.sample_features = sample_features

    def train(self):
        root = self.__split(self.training_set)
        self.__branch_out(root, 1)
        return root

    # define split conditions for each feature, measure importance
    # return a new node: dict with information about split condition or a float in case of a leaf node
    def __split(self, tree_node_records):
        if isinstance(tree_node_records, list) or isinstance(tree_node_records, np.ndarray):
            # choose n features (without replacement)
            random_features = np.random.choice(np.arange(0, self.training_set.shape[1]-1), self.sample_features, replace=False)
            # split for each of the chosen features: dict candidates {i: branches}
            candidates = {}
            for i in random_features:
                condition = np.nanmean(np.array(tree_node_records)[:, i].astype('float'))
                left_branch = []
                right_branch = []
                for record in tree_node_records:
                    if record[i] < condition:
                        left_branch.append(record)
                    else:
                        right_branch.append(record)
                feature_list = list(np.array(tree_node_records)[:, i].astype('float'))
                count = feature_list.count(condition)
                if count == len(feature_list) or len(left_branch) == 0 or len(right_branch) == 0:
                    pass
                else:
                    candidates[i] = list([left_branch, right_branch])
            # in none of the selected features offers a statistically significant split, make a leaf node
            if len(candidates.items()) == 0:
                new_node = np.nanmean(np.array(tree_node_records)[:, -1].astype('float'))
                return new_node
            else:
                # compare importance of the features, choose one that gives lowest pooled sd of children node
                sds = {}
                for k , v in candidates.items():
                    sds[k] = self.__get_feature_importance(v)
                chosen_feature = min(sds.items(), key=lambda x: x[1])
                new_node = {}
                branches = {}
                branches["left"] = candidates[chosen_feature[0]][0]
                branches["right"] = candidates[chosen_feature[0]][1]
                new_node["branches"] = branches
                new_node["feature"] = chosen_feature[0] # index of the chosen feature
                new_node["condition"] = np.nanmean(np.array(tree_node_records)[:, chosen_feature[0]].astype('float'))
                return new_node
                
    def __branch_out(self, tree_node, depths):
        if isinstance(tree_node, dict):
            for side, branch in tree_node["branches"].items():
                if depths >= self.tree_depth or np.array(branch).shape[0] <= self.min_samples:      
                    # take mean value of results (y) for this leaf, converting to floats first
                    branch = np.nanmean(np.array(branch)[:, -1].astype('float'))
                    tree_node[side] = branch
                # else make a node and branch out
                else:
                    branch = self.__split(branch)
                    self.__branch_out(branch, depths+1)
                    tree_node[side] = branch

    def __get_feature_importance(self, branches):
        # POOLED STANDARD DEVIATION
        if len(branches) == 2:
            sd1, sd2 = (0, 0)
            if len(branches[0]) > 0:
                left = np.array(branches[0])[:,-1]
                sd1 = np.std(left)
            if len(branches[1]) > 0:
                right = np.array(branches[1])[:,-1]
                sd2 = np.std(right)
            pooled = math.sqrt((sd1**2 + sd2**2)/2)
            return pooled


    def __predict(self, tree_node, datarow):
        if datarow[tree_node["feature"]] < tree_node["condition"]:
            if isinstance(tree_node["left"], dict):
                return self.__predict(tree_node["left"], datarow)
            else:
                return tree_node["left"]                
        else:
            if isinstance(tree_node["right"], dict):
                return self.__predict(tree_node["right"], datarow)
            else:
                return tree_node["right"]
    
    def get_predictions(self, tree):
        test = self.testing_set
        predicted_values = list()
        for row in test:
            datarow = row[3:-1]
            predicted = self.__predict(tree, datarow)
            predicted_values.append(predicted)
        return predicted_values


class MRRFR(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol    
	
    def configure_args(self):
        super(MRRFR, self).configure_args()
        self.add_passthru_arg('--tree_depth', type=int, default=1000, help='Max depth of decision trees, defaut - 1000')
        self.add_passthru_arg('--min_samples', type=int, default=5, help='Max number of values after a split to make a leaf node')
        # TODO: add option to predict for a single country

    def mapper_init(self):
        self.training = pd.read_csv("rf_training.csv").to_numpy()
        self.testing = pd.read_csv("rf_testing.csv").to_numpy()

    # for each line in the starter file build a tree
    def mapper(self, _, value):
        training = np.delete(self.training, obj = [0, 1, 2], axis = 1)	    
        records_total = training.shape[0]
        features_total = training.shape[1] - 1
        # number of features to consider at a new split
        sample_features = int(math.floor(math.sqrt(features_total)))
        # bootstrap sample records with replacement
        samples = np.random.choice(records_total, records_total)
        bootstrap_samples = training[samples, :]
        tree = RFTree(training = bootstrap_samples, testing = self.testing, depth = self.options.tree_depth, min_samples = self.options.min_samples, sample_features = sample_features)
        model = tree.train()
        predicted = tree.get_predictions(model)
        yield "results", predicted

    def reducer_init(self):
        self.dates_locations = pd.read_csv("rf_testing.csv").to_numpy()[:,[1, 2, -1]]

    def reducer(self, key, values):
        combined_predictions = np.array(list(values))
        if combined_predictions.shape[1] == self.dates_locations.shape[0]:
            result = np.mean(combined_predictions, axis=0)
            result = np.atleast_2d(result).T

            with_date = np.append(self.dates_locations, np.array(result), 1)
            for row in with_date:
                yield None, "{},{},{},{}".format(row[0], row[1], row[2], int(row[3]))

if __name__ == '__main__':
    start = datetime.datetime.now()
    MRRFR.run()
    stop = datetime.datetime.now()
    sys.stderr.write("RUNNING TIME: " + str(stop - start))
