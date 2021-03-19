# Scalable ML algorithms for Hadoop and Spark clusters

## 1. Preprocessing
- filtering out irrelevant rows
- filtering out irrelevant columns
- checking for null values
- replacing null values from other sources or with avg

## 2. Clustering

### K-means

[With Mapreduce](https://slogix.in/how-to-implement-k-means-clustering-using-mapreduce#Source-code)

[Some general info for Hadoop and Spark](http://vargas-solar.com/big-data-analytics/hands-on/k-means-with-spark-hadoop/)

[With Spark](https://www.bmc.com/blogs/python-spark-k-means-example/)

[A deeper guide to K-Means on Spark](https://rsandstroem.github.io/sparkkmeans.html)


### Agglomerative

No ready implementations for MRjob or Spark, has to be implemented from scratch and not always scalable - depending on the dataset.

## 3. Regressions

### LSTM

Deep learning on Hadoop and Spark is a very new topic. Has very few articles and examples.

[WND-LSTM on Mapreduce](https://www.researchgate.net/publication/342641842_A_distributed_WND-LSTM_model_on_MapReduce_for_short-term_traffic_flow_prediction)

[LSTM on Spark with Tensorflow](https://www.slideshare.net/emanueldinardo/distributed-implementation-of-a-lstm-on-spark-and-tensorflow-69787635)

### Polynomial Regression

Spark only has ready libs for linear regression. 

### Support Vector Machine

Very few examples, but they exist

[Article about SVM on Spark](https://ieeexplore.ieee.org/document/7840691)

### Random Forest Regression

Very scalable, well documented and represented, many examples.

[Technical tutoria;: random forest models with Python and Spark](https://www.silect.is/blog/random-forest-models-in-spark-ml/)

[Random Forest classifier with Spark](https://medium.com/rahasak/random-forest-classifier-with-apache-spark-c63b4a23a7cc)

... and many more.

### Gradient Boosted Trees

Again well documented, but the official library has (or had) performance issues.

There is an [alternative lib for LightGBM](https://github.com/Azure/mmlspark)

Other articles

[GBM for Python and Spark](https://ranjithmenon.medium.com/gradient-boosted-tree-regression-spark-dd5ac316a252)

[Example application with code snippets](https://medium.com/@aieeshashafique/gradient-boost-model-using-pyspark-mllib-solving-a-chronic-kidney-disease-problem-13039b6dc099)

