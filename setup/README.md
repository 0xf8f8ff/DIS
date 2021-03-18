# Cluster setup

Run `setup.sh master` to setup Hadoop on a new cluster. Don't run this script on a running cluster.

Before running, check that IPs of all nodes are correct, they would changes if nodes were removed and created again.

If the script runs successfully, it is safe to start dfs and yarn. Run following commands:

- `sh /usr/local/hadoop/sbin/start-dfs.sh`
- `sh /usr/local/hadoop/sbin/start-yarn.sh`

Don't use `stop-all` and `start-all` scripts, they are deprecated and results are sometimes unpredictable.

To check if the setup was correct, 

1. Run `jps`. It must return:

- on master node
```
JPS
NameNode
SecondaryNameNode
ResourceManager
```

- on slave nodes
```
JPS
DataNode
NodeManager
```
2. Run `hdfs dfsadmin -report`. Report must show all three slave nodes.