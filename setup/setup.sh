#!/bin/bash

# run this script on the master node
# all ssh keys have to be added to the 
# .ssh/authorized_hosts file before running
# this script

if [ $# -ne 1 ]; then
    echo -e "Please specify the role of the node: master or slave";
    exit;
fi

nodetype=$1

# 1. Install java and python
echo "Installing Java..."
sudo apt update

if ! sudo apt install openjdk-8-jdk; then
	echo "Error installing Java"
	exit 1
fi
echo "Installed Java, current version: "
java -version

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]; then
    echo "Python is not installed" 
    echo "Installing Python..."
	apt install python3-pip
fi
echo "Installed Python. Default Python version now:"
python --version

pip install mrjob

# Install python on all nodes
if [ $nodetype = "master" ]; then
	ssh -t ubuntu@slave-1 "apt install python3-pip"
	ssh -t ubuntu@slave-2 "apt install python3-pip"
	ssh -t ubuntu@slave-3 "apt install python3-pip"
fi

# 2. Download hadoop on master
if [ $nodetype = "master" ]; then
	echo "Downloading Hadoop 3.2.1"
	if ! sudo wget -P ~ https://mirrors.sonic.net/apache/hadoop/common/hadoop-3.2.1/hadoop-3.2.1.tar.gz; then
	echo "Error downloading Hadoop"
	exit 1
	fi
fi

# 3. Set up hostnames and IPs of all nodes
# Note: change hostnames and IPs here according to the actual topology
echo "Updating IPs of all nodes"
sudo tee -a /etc/hosts << EOF
10.10.1.35 master
10.10.1.112 slave-1
10.10.1.16 slave-2
10.10.1.33 slave-3
EOF

if [ $nodetype = "master" ]; then
	ssh -t ubuntu@slave-1 "sudo rm /etc/hosts"
	scp /etc/hosts slave-1:/etc/
	ssh -t ubuntu@slave-2 "sudo rm /etc/hosts"
	scp /etc/hosts slave-2:/etc/
	ssh -t ubuntu@slave-3 "sudo rm /etc/hosts"
	scp /etc/hosts slave-3:/etc/
fi

# 4. Copy Hadoop image to all slave nodes
# Note: change hostnames if necessary
if [ $nodetype = "master" ]; then
	echo "Copying Hadoop 3.2.1 to all slave nodes"
	scp hadoop-3.2.1.tar.gz slave-1:/home/ubuntu/
	scp hadoop-3.2.1.tar.gz slave-2:/home/ubuntu/
	scp hadoop-3.2.1.tar.gz slave-3:/home/ubuntu/
fi


# 5. Unpack hadoop, set up java path
echo "Unpacking Hadoop..."
tar xzf hadoop-3.2.1.tar.gz
mv hadoop-3.2.1 hadoop

echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" | tee -a ~/hadoop/etc/hadoop/hadoop-env.sh
sudo mv hadoop /usr/local/hadoop


if [ $nodetype = "master" ]; then
	echo "Unpacking Hadoop on slave nodes..."
	ssh -t ubuntu@slave-1 "tar xzf hadoop-3.2.1.tar.gz"
	ssh -t ubuntu@slave-1 "mv hadoop-3.2.1 hadoop"
	ssh -t ubuntu@slave-1 "sudo mv hadoop /usr/local/hadoop"
	ssh -t ubuntu@slave-1 "tar xzf hadoop-3.2.1.tar.gz"
	ssh -t ubuntu@slave-1 "mv hadoop-3.2.1 hadoop"
	ssh -t ubuntu@slave-1 "sudo mv hadoop /usr/local/hadoop"
	ssh -t ubuntu@slave-1 "tar xzf hadoop-3.2.1.tar.gz"
	ssh -t ubuntu@slave-1 "mv hadoop-3.2.1 hadoop"
	ssh -t ubuntu@slave-1 "sudo mv hadoop /usr/local/hadoop"

fi

echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/hadoop/bin:/usr/local/hadoop/sbin"JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre"' | sudo tee /etc/environment 

source /etc/environment
if [[ ! -v JAVA_HOME ]]; then
	echo "Error setting PATH variable"
    echo "Check /etc/environment for any issues"
	exit 1
fi

sudo tee -a ~/.bashrc << EOF
export HADOOP_HOME="/usr/local/hadoop"
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export HADOOP_HDFS_HOME=$HADOOP_HOME
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_YARN_HOME=$HADOOP_HOME
EOF

if [ $nodetype = "slave" ]
	echo "Slave configuration done"
	exit 1
fi

# 6. Configure Hadoop on master
echo "Updating Hadoop configuration files"
sudo tee /usr/local/hadoop/etc/hadoop/core-site.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
	<property>
	<name>fs.defaultFS</name>
	<value>hdfs://master:9000</value>
	</property>
</configuration>
EOF

sudo tee  /usr/local/hadoop/etc/hadoop/hdfs-site.xml << EOF
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
	<property>
	<name>dfs.namenode.name.dir</name><value>/usr/local/hadoop/data/nameNode</value>
	</property>
	<property>
	<name>dfs.datanode.data.dir</name><value>/usr/local/hadoop/data/dataNode</value>
	</property>
	<property>
	<name>dfs.replication</name>
	<value>3</value>
	</property>
</configuration>
EOF

sudo tee /usr/local/hadoop/etc/hadoop/mapred-site.xml << EOF
<?xml version="1.0"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
        <property>
           <name>mapreduce.framework.name</name>
           <value>yarn</value>
        </property>
        <property>
           <name>yarn.app.mapreduce.am.env</name>
           <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
        </property>
        <property>
           <name>mapreduce.map.env</name>
           <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
        </property>
        <property>
           <name>mapreduce.reduce.env</name>
           <value>HADOOP_MAPRED_HOME=$HADOOP_HOME</value>
        </property>
</configuration>
EOF

sudo tee sudo tee /usr/local/hadoop/etc/hadoop/yarn-site.xml << EOF
<?xml version="1.0"?>
<configuration>
    <property>
		<name>yarn.acl.enable</name>
		<value>0</value>
    </property>

    <property>
		<name>yarn.resourcemanager.hostname</name>
		<value>master</value>
    </property>

    <property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
    </property>
    <property>
		<name>yarn.nodemanager.aux-services.mapreduce_shuffle.class</name>
		<value>org.apache.hadoop.mapred.ShuffleHandler</value>
    </property>
</configuration>
EOF

sudo tee /usr/local/hadoop/etc/hadoop/workers << EOF
slave-1
slave-2
slave-3
EOF

# 7. Copy configuration to slave nodes
echo "Copying hadoop configuration to slave nodes"
scp /usr/local/hadoop/etc/hadoop/* slave-1:/usr/local/hadoop/etc/hadoop/
scp /usr/local/hadoop/etc/hadoop/* slave-2:/usr/local/hadoop/etc/hadoop/
scp /usr/local/hadoop/etc/hadoop/* slave-3:/usr/local/hadoop/etc/hadoop/

echo "Formatting namenode"
source /etc/environment
if ! hdfs namenode -format; then
	echo "Error formatting namenode"
	exit 1
fi

echo "Hadoop setup complete"
echo "It is safe to start dfs and yarn now"

