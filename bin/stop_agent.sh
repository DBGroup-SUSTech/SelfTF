#!/bin/bash
source $(dirname "$0")/common.sh

cd $SELFTF_HOME

pid_file=agent.pid

pid=$(cat $pid_file)

kill $pid
killall -9 $SCRIPT_PYTHON_EXECUTABLE
rm $pid_file

echo `hostname` agent PID:$pid  is killed

