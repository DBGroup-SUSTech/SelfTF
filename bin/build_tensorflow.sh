SOURCE_PATH=/root/tensorflow
SELFTF_PATH=/root/pstuner
cd $SOURCE_PATH

rm -rf /tmp/tensorflow*.whl

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/

WHL=$(ls /tmp/tensorflow*.whl)

cat $SELFTF_PATH/slaves | xargs -i -P40 scp $WHL {}:
cat $SELFTF_PATH/slaves | xargs -i -P40 ssh {} bash -c 'pip install --upgrade `ls /root/pstuner tensorflow*.whl`'
