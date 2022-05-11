export DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`

export SELFTF_MASTER_NODE="ssd02"
export AMPQ_MASTER_NODE=$SELFTF_MASTER_NODE
export HDFS_MASTER_NODE=$SELFTF_MASTER_NODE
export SELFTF_HOME=/root/pstuner
export SELFTF_HDFS_HOME="hdfs://$HDFS_MASTER_NODE:8020/user/root/pstuner"
export DATASET_BASE_PATH="hdfs://$HDFS_MASTER_NODE:8020/user/root/train_data"
export SCRIPT_PYTHON_EXECUTABLE="/root/anaconda2/bin/python"

export PYTHONPATH=$PYTHONPATH:$SELFTF_HOME
export SELFTF_NUM_COMPUTE_NODE=$(wc -l < $SELFTF_HOME/slaves)
export SELFTF_NUM_THREAD=$(grep -c ^processor /proc/cpuinfo)