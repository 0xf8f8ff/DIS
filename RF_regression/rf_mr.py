from mrjob.job import MRJob
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
        print("TYPE OF TRAINING SET: ", type(self.training_set))
        root = self.__split(self.training_set)
        print("Type of root: ", type(root))
        self.__branch_out(root, 1)
        return root

    # define split conditions for each feature, measure importance
    # return a new internal node with condition and branches
    def __split(self, tree_node_records):
        if isinstance(tree_node_records, list) or isinstance(tree_node_records, np.ndarray):
            print("RIGHT TYPE, MAKING NODE: ", type(tree_node_records))
            # choose n features (without replacement)
            print("Total feature columns: ", self.training_set.shape[1] - 1)
            random_features = np.random.choice(np.arange(0, self.training_set.shape[1]-1), self.sample_features, replace=False)
    #        print("Features chosen: " + str(random_features))
            # split on each {i: branches}
            candidates = {}
            for i in random_features:
                condition = np.nanmean(np.array(tree_node_records)[:, i].astype('float'))
    #            print("Feature mean for feature " + str(i) + ": " + str(condition))
                left_branch = []
                right_branch = []
                for record in tree_node_records:
                    if record[i] < condition:
                        left_branch.append(record)
                    else:
                        right_branch.append(record)
    #            print("Branches len: " + str([len(left_branch), len(right_branch)]))
                feature_list = list(np.array(tree_node_records)[:, i].astype('float'))
                count = feature_list.count(condition)
                if count == len(feature_list) or len(left_branch) == 0 or len(right_branch) == 0:
                    print("Feature ", i, " has only one value: ", condition)
                else:
                    candidates[i] = list([left_branch, right_branch])
    #        print("Candidates chosen: " + str(candidates.keys()))
            if len(candidates.items()) == 0:
                # make a leaf node
                new_node = np.nanmean(np.array(tree_node_records)[:, -1].astype('float'))
                print("No candidates, LEAF NODE: ", type(new_node))
                return new_node
            else:
                # compare importance and choose one
                sds = {}
                for k , v in candidates.items():
                    sds[k] = self.__get_feature_importance(v)
                chosen_feature = min(sds.items(), key=lambda x: x[1])
    #            print("CHOSEN FEATURE: " + str(chosen_feature[0]))
                new_node = {}
                branches = {}
                branches["left"] = candidates[chosen_feature[0]][0]
                branches["right"] = candidates[chosen_feature[0]][1]
                new_node["branches"] = branches
                new_node["feature"] = chosen_feature[0] # index of the chosen feature
                new_node["condition"] = np.nanmean(np.array(tree_node_records)[:, chosen_feature[0]].astype('float'))
                return new_node
        else:
            print("WRONG TYPE OF NODE: ", type(tree_node_records))

    def __branch_out(self, tree_node, depths):
        
        print("Branching out, depth: " + str(depths))
        print("Type of tree_node: ", type(tree_node))
        if isinstance(tree_node, dict):
            for side, branch in tree_node["branches"].items():
                #print("BRANCH: ", branch)
                if depths >= self.tree_depth or np.array(branch).shape[0] <= self.min_samples:
                    
                    # take mean value of results (y) for this leaf, converting to floats first
                    branch = np.nanmean(np.array(branch)[:, -1].astype('float'))
                    tree_node[side] = branch
                    print("LEAF NODE: ", branch)
                # else make a node and branch out on a condition
                else:
                    branch = self.__split(branch)
                    self.__branch_out(branch, depths+1)
                    tree_node[side] = branch

        elif isinstance(tree_node, list):
            print("NODE IS LIST OF LEN ", len(tree_node))
        
    def __get_feature_importance(self, branches):
        # POOLED STANDARD DEVIATION
        if len(branches) != 2:
            return 666
        sd1, sd2 = (0, 0)
        if len(branches[0]) > 0:
            left = np.array(branches[0])[:,-1]
            sd1 = np.std(left)
        if len(branches[1]) > 0:
            right = np.array(branches[1])[:,-1]
            sd2 = np.std(right)

        pooled = math.sqrt((math.sqrt(sd1) + math.sqrt(sd2))/2)
        return pooled


    def __predict(self, tree_node, datarow):
        print("Predicting, type of tree_node: ", type(tree_node))
        print("Left type: ", type(tree_node["left"]))
        print("Right type: ", type(tree_node["right"]))
        #if isinstance(tree_node, dict):
        if datarow[tree_node["feature"]] < tree_node["condition"]:
            if isinstance(tree_node["left"], dict):
                return self.__predict(tree_node["left"], datarow)
            else:
                print("PREDICTING: ", tree_node["left"])
                return tree_node["left"] # return left branch as prediction
                
        else:
            if isinstance(tree_node["right"], dict):
                return self.__predict(tree_node["right"], datarow)
            else:
                print("PREDICTING: ", tree_node["right"])
                return tree_node["right"] # return right value
    
    def get_predictions(self, tree):
        test = self.testing_set
        predicted_values = list()
        for row in test:
            oldrow = row
            #print("Row before: ", len(oldrow))
            datarow = row[3:-1]
            #print("Row after: ", len(datarow))
            predicted = self.__predict(tree, datarow)
            print("Predicted value: ", predicted)
            print("Real value: ", oldrow[-1])
            #date_real_predicted = list(row[0], predicted)
            #predicted_values.append(date_real_predicted)
            predicted_values.append(predicted)
        return predicted_values


class MRRFR(MRJob):
    def configure_args(self):
        super(MRRFR, self).configure_args()
        #self.add_passthru_arg('--tree_nr', type=int, default=100, help='Number of decision trees to build, default - 100')
        self.add_passthru_arg('--tree_depth', type=int, default=10, help='Max depth of decision trees, defaut - 10')
        self.add_passthru_arg('--min_samples', type=int, default=10, help='Max number of values after a split to make a leaf node')
        self.add_passthru_arg('--location', type=str, default="Norway", help='Country name, default - Norway')
        # TODO: add option to process all countries


    def mapper_init(self):
        self.training = pd.read_csv("rf_training.csv")
        self.testing = pd.read_csv("rf_testing.csv")

    def mapper(self, _, next_tree):
        records_total = self.training.shape[0]
        features_total = self.training.shape[1] - 1
        sample_features = int(math.floor(math.sqrt(features_total)))
        # pick sample records with replacement
        samples = np.random.choice(records_total, records_total)
        tree_samples = self.training.ix[samples]
        tree = RFTree(training = tree_samples, testing = self.testing, depth = self.options.tree_depth, min_samples = self.options.min_samples, sample_features = sample_features)
        model = tree.train()
        predicted = tree.get_predictions(model)
        yield None, predicted
    
    def reducer(self, key, values):
        date_column = self.testing[:, 2]
        location_column = self.testing[:, 1]

        combined_predictions = np.array(list(values))
        if len(date_column) != combined_predictions.shape[0]:
            yield None, "LENGTH MISMATCH: " + len(date_column) + " vs " + combined_predictions.shape[0]
        else:
            result = np.mean(combined_predictions, axis=1)
            with_date = np.vstack(location_column, date_column, result)
            yield None, with_date

if __name__ == '__main__':
    start = datetime.datetime.now()
    MRRFR.run()
    stop = datetime.datetime.now()
    sys.stderr.write("RUNNING TIME: " + str(stop - start))