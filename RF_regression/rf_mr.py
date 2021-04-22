from mrjob.job import MRJob
from mrjob.step import MRStep
import math
import numpy as np

class TreeNode:
    branches: list # list of children
    feature: int
    split: float
    leaf_value: float # final value


class RFTree:
    def __init__(self, training, testing, depth, min_samples, sample_features):
        self.training_set = np.array(training)
        self.testing_set = np.array(testing)
        self.tree_depth = depth
        self.min_samples = min_samples
        self.sample_features = sample_features

    def model(self):
        root = self.training_set
        self.branch_out(root, 1)
        return root

    # define split conditions for each feature, measure importance
    # return a new internal node with condition and branches
    def split(self, tree_node):
        # choose n features
        random_features = np.random.choice(self.training_set.shape[0]-1, self.sample_features)
        # split on each {i: branches}
        candidates = {}
        for i in random_features:
            condition = np.mean(self.training_set[:, i].astype('float'))
            left_branch = []
            right_branch = []
            for record in tree_node:
                if record[i] <= condition:
                    left_branch.append(record)
                else:
                    right_branch.append(record)
            candidates[i] = [left_branch, right_branch]

        # compare importance and choose one
        sds = {}
        for k , v in candidates:
            sds[k] = self.get_feature_importance(v)
        chosen_feature = min(sds.items(), key=lambda x: x[1])
        tree_node["branches"] = candidates[chosen_feature[0]]
        tree_node["feature"] = chosen_feature[0] # index of the chosen feature
        tree_node["condition"] = np.mean(self.training_set[:, chosen_feature[0]].astype('float'))

    def branch_out(self, tree_node, depths):
        # shape = 2 for all features except the seasonal (where it is 4)
        for branch in tree_node["branches"]:
            if depths >= self.tree_depth or branch.shape[0] <= self.min_samples:
                # take mean value of results (y) for this leaf, converting to floats first
                branch = np.mean(branch[:, -1].astype('float'))
            # else make a node and branch out on a condition
            else:
                branch = self.split(branch)
                self.branch_out(branch, depths+1)
    
    def get_feature_importance(self, branches):
        # POOLED STANDARD DEVIATION
        if len(branches) != 2:
            return 0
        sd1 = np.std(branches[0][:,-1])
        sd2 = np.std(branches[1][:,-1])
        pooled = math.sqrt((math.sqrt(sd1) + math.sqrt(sd2))/2)
        return pooled


    def train(self, tree_node, datarow):
        # TODO: split based on attributes
        # split further if not leaf node
        pass

    def predict(self, tree):
        test = self.testing_set
        predicted_values = []
        for row in test:
            predicted = self.train(tree, row)
            date_real_predicted = list(row[0], row[-1], predicted)
            predicted_values.append(date_real_predicted)
        return predicted_values

class MRRandomForestRegression(MRJob):
    def configure_args(self):
        super(MRRandomForestRegression, self).configure_args()
        self.add_passthru_arg('--tree_nr', type=int, default=100, help='Number of decision trees to build, default - 100')
        self.add_passthru_arg('--tree_depth', type=int, default=10, help='Max depth of decision trees, defaut - 10')
        self.add_passthru_arg('--min_samples', type=int, default=10, help='Max number of values after a split to make a leaf node')
        self.add_passthru_arg('--location', type=str, default="Norway", help='Country name, default - Norway')

    def steps(self):
        return [
            MRStep(mapper_init = self.mapper_init_preprocess,
            mapper = self.mapper_preprocess),
            MRStep(mapper_init = self.mapper_init_model,
            reducer = self.reducer_model)
        ]

    def mapper_init_preprocess(self):
        header_names = ["index","location","date","total_cases","new_cases","new_cases_smoothed","total_deaths","new_deaths","new_deaths_smoothed","total_cases_per_million","new_cases_smoothed_per_million","total_deaths_per_million","new_deaths_smoothed_per_million","reproduction_rate","total_tests","total_vaccinations","people_vaccinated","people_fully_vaccinated","new_vaccinations","new_vaccinations_smoothed","total_vaccinations_per_hundred","people_vaccinated_per_hundred","people_fully_vaccinated_per_hundred","new_vaccinations_smoothed_per_million","stringency_index","population","population_density","median_age","aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","cardiovasc_death_rate","diabetes_prevalence","female_smokers","male_smokers","hospital_beds_per_thousand","life_expectancy","human_development_index","new_tests","days_after_100_cases","internal_flights","international_arrivals"]
        # make dict [header]index for quick lookup
        self.headers = {}
        for i, h in header_names:
            self.headers[h] = i

        # one-hot encode date as season
        #season_names = pd.Series(["winter", "spring", "summer", "autumn"])
        # winter = [1,0,0,0]
        # spring = [0,1,0,0]
        # summer = [0,0,1,0]
        # autumn = [0,0,0,1]
        
        # yearSeasons = [
        #     winter, winter, spring,
        #     spring, spring, summer,
        #     summer, summer, autumn,
        #     autumn, autumn, winter,
        # ]

        # self.seasons = dict(zip(range(1,13), yearSeasons))

        # initialize structures for training and testing subsets 
        self.training = []
        self.testing = []
                
    def mapper_preprocess(self, _, line):
        row = line.split(",")
        # ignore malformed rows, only consider rows for the chosen country
        if len(row) == len(self.headers) and row[1] == self.options.location:
            # choose only independent features
            new_record = [row["date"], row["total_cases_per_million"], row["total_deaths_per_million"], row["reproduction_rate"], row["stringency_index"], row["population_density"], row["gdp_per_capita"], row["median_age"], row["human_development_index"], row["hospital_beds_per_thousand"], row["diabetes_prevalence"], row["days_after_100_cases"], row["internal_flights"], row["international_arrivals"], row["new_cases"]]
            # replace all NaN values with zeroes
            new_record = np.nan_to_num(new_record)
            # replace date string with month number
            #new_record = list(new_record)
            date = new_record[0]
            parts = date.split("-")
            if len(parts) == 3:
                # make a column with month numbers
                encoded_date = parts[1] #self.seasons[int(parts[1])]
                # for training data (year 2020) append the column with new cases
                # then add column with one-hot encoded season
                new_record.insert(1, encoded_date)
                if parts[0] == "2021":
                    self.testing.append(list(new_record))
                else:
                    self.training.append(list(new_record))
    

    def mapper_init_model(self):
        self.total_features = self.training_data.shape[1]
        self.sample_features = math.floor(math.sqrt(self.total_features))

    def mapper_model(self, _):
        # TODO: leave or remove the raw date column?
        training_data = np.array(self.training)#[1:]
        length = training_data.shape[0]

        # bootstrap-sample trees with replacement
        for i in range(1, self.options.trees):
            # pick random 4 features for the tree
            #cols = np.random.choice(self.total_features, self.sample_features)
            #tree_data = training_data[:, cols]
            # pick sample records with replacement
            samples = np.random.choice(length, length)
            tree_samples = training_data[samples]
            tree = RFTree(training = tree_samples, testing = self.testing, depth = self.options.tree_depth, min_samples = self.options.min_samples, sample_features = self.sample_features)
            tree_model = tree.model()
            predicted = tree.predict(tree_model)
            yield None, predicted

    def reducer_model(self, _, values):
        # TODO: sort values by date

        # TODO: take mean for each date as final prediction
        all_predictions = np.array(list(values))
        # TODO (MAYBE) transform into rows [date, all predictions for this date] - needs an extra combiner?
