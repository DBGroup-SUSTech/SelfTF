import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework

from selftf.lib import tf_program_util
from selftf.tf_job.classification.ml_model import SVMModel_with_linear, LogisticRegressionModel
from selftf.tf_job.classification.read_libsvm_data import read_batch


class SVM_LR(MLJobFramework):

    def __init__(self):
        MLJobFramework.__init__(self)
        self.train_data_batch_tensor = None
        self.num_features = 0

        self.y = None
        self.sp_indices = None
        self.shape = None
        self.ids_val = None
        self.weights_val = None

        self.observe_loss = None

        # for dataset
        tf.app.flags.DEFINE_string("data_dir", "", "")

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        FLAGS = context.get_tf_flag()

        self.num_features = FLAGS.num_Features

        # count the number of global steps
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False)
        context.set_global_step(global_step)

        # inout data
        path = context.get_tf_flag().data_dir


        trainset_files = map(lambda x: path+"/"+ x, tf.gfile.ListDirectory(path))
        train_filename_queue = tf.train.string_input_producer(trainset_files)
        train_reader = tf.TextLineReader()
        key_tensor, line_tensor = train_reader.read(train_filename_queue)
        self.train_data_batch_tensor = tf.train.shuffle_batch(
            [line_tensor],
            batch_size=context.get_batch_size(),
            capacity=100,
            min_after_dequeue=50
        )

        with tf.variable_scope('placeholder'):
            if FLAGS.ML_model == "SVM":
                y_shape = 1
            else:
                y_shape = 2
            self.y = tf.placeholder(tf.float32, [None, y_shape])
            self.sp_indices = tf.placeholder(tf.int64, name="sp_indices")
            self.shape = tf.placeholder(tf.int64, name="shape")
            self.ids_val = tf.placeholder(tf.int64, name="ids_val")
            self.weights_val = tf.placeholder(tf.float32, name="weights_val")

        with tf.variable_scope('parameter'):
            x_data = tf.SparseTensor(self.sp_indices, self.weights_val, self.shape)
            # x_data_SVM = tf.sparse_to_den        se(sp_indices, shape, weights_val)

        with tf.variable_scope('loss'):
            if FLAGS.ML_model == "SVM":
                SVM_loss = SVMModel_with_linear(x_data, self.y, self.num_features)
                self.observe_loss = SVM_loss
            else:
                LR_loss, LR_loss_l2 = LogisticRegressionModel(x_data, self.y, self.num_features)
                self.observe_loss = LR_loss

        # specify optimizer
        if FLAGS.ML_model == "SVM":
            # LR_train_op = grad_op.minimize(LR_loss_l2, global_step=global_step)
            context.set_train_op(SVM_loss)
        else:
            # SVM_train_op = grad_op.minimize(SVM_loss, global_step=global_step)
            context.set_train_op(LR_loss_l2)



    def get_feed_dict(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        sess = self.sess
        batch_size = context.get_batch_size()
        label_one_hot, label, indices, sparse_indices, weight_list, read_count = read_batch(sess, self.train_data_batch_tensor,
                                                                                         batch_size)
        if context.get_tf_flag().ML_model == "SVM":
            return {self.y: label,
                self.sp_indices: sparse_indices,
                self.shape: [read_count,
                         self.num_features],
                self.ids_val: indices,
                self.weights_val: weight_list}
        else:
            return {self.y: label_one_hot,
                self.sp_indices: sparse_indices,
                self.shape: [read_count,
                         self.num_features],
                self.ids_val: indices,
                self.weights_val: weight_list}

    def get_observe_loss_variable(self, context):
        return self.observe_loss


model = SVM_LR()
model.run()
