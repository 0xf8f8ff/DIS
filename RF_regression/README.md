# Random Forest regression

**rf_preprocess.py**: data preparation script, writes testing and training datasets in csv format to `rf_training.csv` and `rf_testing.csv`.

**make_dummy.py**: a script for creating a dummy file, pass the desired number of decision trees as a command line argument. Example usage: `python make_dummy.py 1000` for running the regression with 1000 trees.

**rf_mr.py**: *mrjob* MapReduce implementation of the Random Forest regression algorithm. Options: *--tree-depth* - to limit the depth of the decision trees, *--min-samples* - the number of records in a node to stop splitting and form a leaf node.

Usage: 
```
python rf_mr.py --files rf_testing.csv,rf_training.csv -r hadoop hdfs:///testdata/dummy --output-dir hdfs:///rfregression --no-output
```

**rf_mllib.ipynb**: Random Forest regression with Spark's *mllib*.

**rf_plot.py**: graph plotting script.