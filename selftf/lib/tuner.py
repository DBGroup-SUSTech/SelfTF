import json
import logging
import math
import random
import sys
from abc import abstractmethod

import sklearn
from scipy.optimize import curve_fit

from lhs import LHSAdapter
import common
from selftf.lib.common import PSTunerConfiguration, PSTunerTrainingData
from selftf.lib.gpr.constraints import ParamConstraintHelper
from selftf.lib.gpr.preprocessing import DummyEncoder
from selftf.lib.gpr.gp_tf import *
import numpy as np


# Key of conf_dict to tf_program_util
_ps_num = "ps_num"
_intra_op_parallelism_threads = "intra_op_parallelism_threads"
_inter_op_parallelism_threads = "inter_op_parallelism_threads"
_n_partition = "n_partition"
_learning_rate = "learning_rate"
_batch_size = "batch_size"
_optimizer = "optimizer"
# =======
_sync_protocal = "sync_protocal"
_ps_stragegy = "ps_stragegy"
_session_inter_op_thread_pools = "session_inter_op_thread_pool"
_placement_period = "placement_period"
_allow_soft_placement = "allow_soft_placement"
_operation_timeout_in_ms = "operation_timeout_in_ms"
_do_common_subexpression_elimination = "do_common_subexpression_elimination"
_do_constant_folding = "do_constant_folding"
_do_function_inlining = "do_function_inlining"


_default_func_name="bo"

_average_runtime_sample = 72


class EstimationResult:
    def __init__(self, remaining_iteration,
        remaining_time, average_iteration_time):
        self.remaining_iteration = remaining_iteration
        self.remaining_time = remaining_time
        self.average_iteration_time = average_iteration_time

class PSTunerTrainingAlgorithm(object):

    @abstractmethod
    def train(self, training_data):
        pass

    @abstractmethod
    def get_best_config(self):
        """
        :param common.Job job_obj:
        :return:
        """
        pass


class GPTrainingModel(PSTunerTrainingAlgorithm):

    def __init__(self, tf_config_util, epsilon=0.5, func_name=_default_func_name):
        """
        :param TFConfigUtil tf_config_util:
        """
        self.model = None
        self.tf_config_util = tf_config_util

        self.target_loss = epsilon # target loss
        self.max_loss = 0.0 #max loss

        self.func_name = func_name

        self.normalized_training_data = None
        self.training_data_remaining_time = None

        self.logger = logging.getLogger(__name__)

    def get_vector_with_predict_runtime(self, training_data):
        training_x, training_y = self.group_training_data_to_list_config_and_output_vector(
            training_data)
        normalized_training_data = self.tf_config_util.normalize_list_config_vector(
            training_x)
        normalized_training_y = self.tf_config_util.normalize_y(training_y)
        return normalized_training_data, normalized_training_y

    def train_gpr(self, normalized_training_data, normalized_training_y):
        self.normalized_training_data = normalized_training_data
        try:

            MAX_ITER = 500
            MAX_TRAIN_SIZE = 7000
            BATCH_SIZE = 3000
            NUM_THREADS = 4
            DEFAULT_LENGTH_SCALE = 1.0
            DEFAULT_MAGNITUDE = 1.0
            DEFAULT_RIDGE = 1.0
            DEFAULT_LEARNING_RATE = 0.01
            DEFAULT_EPSILON = 1e-6
            DEFAULT_MAX_ITER = 5
            DEFAULT_RIDGE = 1.0
            DEFAULT_SIGMA_MULTIPLIER = 3.0
            DEFAULT_MU_MULTIPLIER = 1.0

            gpr_gd = GPRGD(length_scale=DEFAULT_LENGTH_SCALE,
                      magnitude=DEFAULT_MAGNITUDE,
                      max_train_size=MAX_TRAIN_SIZE,
                      batch_size=BATCH_SIZE,
                      num_threads=NUM_THREADS,
                      learning_rate=DEFAULT_LEARNING_RATE,
                      epsilon=DEFAULT_EPSILON,
                      max_iter=MAX_ITER,
                      sigma_multiplier=DEFAULT_SIGMA_MULTIPLIER,
                      mu_multiplier=DEFAULT_MU_MULTIPLIER)


            minX, maxX = self.tf_config_util.get_normalized_min_max_vectors()

            gpr_gd.fit(normalized_training_data, normalized_training_y,
                       minX, maxX, ridge=DEFAULT_RIDGE) #TODO tune this parameter
            self.model = gpr_gd
        except:
            self.logger.exception("Something wrong with GP training")
            raise
        finally:
            self.logger.debug("Exit the block of GP training")

    def train(self, training_data):
        """
        :param list[selftf.lib.common.PSTunerTrainingData] training_data:
        :return:
        """

        normalized_training_data, normalized_training_y = \
            self.get_vector_with_predict_runtime(training_data)

        self.train_gpr(normalized_training_data, normalized_training_y)
        self.logger.debug("Finish training new GP model")

    def group_training_data_to_list_config_and_output_vector(self, list_tf_config_training_data):
        """
        :param list[selftf.lib.common.PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """

        group_by_ps_config = self.tf_config_util.get_lastest_training_data_group_by_ps_config(list_tf_config_training_data)
        list_config_vector = []
        list_output_vector = []

        for config, group_list_training_data in group_by_ps_config.iteritems():

            last_loss = group_list_training_data[-1].loss
            list_config_vector.append(
                self.tf_config_util.tf_config_to_vector(config) + [last_loss])
            logging.debug("config: %s" % str(config))
            estimation_result= self.estimate_remaining_time(group_list_training_data)
            remaining_time = estimation_result.remaining_time
            if math.isnan(remaining_time):
                remaining_time = sys.float_info.max
            list_output_vector.append(remaining_time)

        logging.debug("Config vector list(x):%s remaining time(y):%s" % (str(list_config_vector), str(list_output_vector)))
        return list_config_vector, list_output_vector

    def estimate_remaining_iteration_preprocess(self, step_sample, loss_sample):
        return  EstimationFuncUtil.outlier_median_discard(step_sample, loss_sample)

    def estimate_remaining_iteration(self, step_sample, loss_sample, target_loss):

        step_sample, loss_sample = self.estimate_remaining_iteration_preprocess(
            step_sample, loss_sample)

        # define bo_fun
        def bo_fun(x, h, d, j0):
            return np.true_divide(h, x) * np.log(np.true_divide(d, x)) + j0

        # wrap bo_fun and do curve fitting
        d_hat = np.amax(loss_sample)
        j0_hat = step_sample[0]

        def wrapped_fun(x, h):
            return bo_fun(x, h, d_hat, j0_hat)

        # a_fit, _ = curve_fit(wrapped_fun, loss_sample, step_sample)
        xx = np.true_divide(1, loss_sample) * np.log(
            np.true_divide(d_hat, loss_sample))
        yy = step_sample - j0_hat
        h_fit = [np.true_divide(np.dot(xx, yy), np.dot(xx, xx))]

        pred_end_step = wrapped_fun(target_loss, *h_fit)
        pred_remain_step = pred_end_step - j0_hat

        # logging.debug(
        #     "alpha:%f, overall_avg_iteration_time: %f, remaining_iteration: %f, remaining_time: %f, estimiate_current_iteration: %f" % (
        #         float(xx),
        #         elapsed_time_in_ms, remaining_iteration,
        #         y, estimiate_current_iteration))

        return pred_remain_step

    def estimate_remaining_time(self, list_training_data):
        """
        :param list[PSTunerTrainingData] list_training_data:
        :return:
        """
        # train the alpha through the training_data
        last_step = list_training_data[-1].step

        step_samples = np.array(
            map(lambda x: x.step, list_training_data))
        loss_samples = np.array(
            map(lambda x: x.loss, list_training_data))

        remaining_iteration = self.estimate_remaining_iteration(
            step_sample=step_samples,
            loss_sample=loss_samples,
            target_loss=self.target_loss
        )

        # n_workers = grouped_training_record.ps_config.worker_num

        elapsed_time_in_ms = self.tf_config_util.list_training_data_get_average_time(
            list_training_data)

        # predict update # of workers
        # grouped_training_record.ps_config.
        y = remaining_iteration * elapsed_time_in_ms

        return EstimationResult(
            remaining_iteration=remaining_iteration,
            remaining_time=y,
            average_iteration_time=elapsed_time_in_ms
        )


    def calc_reamaining_time(self, training_data, alpha):
        """
        :param PSTunerTrainingData training_data:
        :return:
        """
        return (alpha / self.target_loss) * training_data.elapsed_time_in_ms

    def get_best_config(self, sample_size=100, list_existing_conf=[], average_recovery_time=0, full_space=False, last_loss=0.0,job_obj=None):
        """
        :param list[selftf.lib.common.PSTunerConfiguration] list_existing_conf:
        :return:
        """
        # if full_space:
        #     list_config_obj = self.tf_config_util.tf_config_manager.generate_tf_config_by_full_space()
        #     list_config_vector = map(lambda  x: self.tf_config_util.normalize_config_vector(self.tf_config_util.tf_config_to_vector((x)))
        #                              , list_config_obj)
        #     logging.debug("Hack: generate full space in online tuning phase: %s" % json.dumps(list_config_obj,
        #                                                                                       cls=common.PSTunerConfigurationSerializer))
        # else:
        #     list_config_vector = lhs.LHS_samples(len(self.tf_config_util.config_sequence), sample_size)
        #     list_tf_config = map(lambda x: self.tf_config_util.config_vector_to_config_obj(self.tf_config_util.denormalize_config_vector(x)),
        #                          list_config_vector)  # just for logging
        #     logging.debug("Online tuning, generateed normalized LHS config: %s " % json.dumps(list_tf_config,
        #                                                                                       cls=common.PSTunerConfigurationSerializer))
        # list_ypreds = []
        # for config_vector in list_config_vector:
        #     try:
        #         logging.debug("Testing config vector: %s " % config_vector)
        #         gprresult = self.model.predict(np.array([config_vector]))
        #         list_ypreds.append(float(gprresult.ypreds[0]))
        #     except:
        #         logging.exception("Same config exist ignore it")
        #         list_ypreds.append(sys.float_info.max - 10000.0)
        #         continue
        # # list_ypreds = map(lambda x: self.model.predict(np.array([x])).ypreds[0], list_config_vector)
        #
        # # adjust time with recovery time, the last one should be the current vector
        # # list_ypreds = gpr_gdresult.ypreds
        # list_config_vector = np.concatenate((list_config_vector, np.array(self.normalized_training_data)))
        # list_ypreds = np.concatenate((np.array(list_ypreds), np.array(self.training_data_remaining_time)))
        #
        # min_value = list_ypreds[-1]
        # min_idx = len(list_ypreds) - 1
        # for x in range(0, len(list_ypreds) - 1):
        #     adjusted_ypreds = list_ypreds[x] + average_recovery_time
        #     logging.debug("ypreds:%f recovery_time:%f adjusted_ypreds:%f" % (
        #     list_ypreds[x], average_recovery_time, adjusted_ypreds))
        #     if adjusted_ypreds < min_value:
        #         logging.debug("Found new min ypreds: org: %f, new_min: %f" % (min_value, adjusted_ypreds))
        #         min_value = adjusted_ypreds
        #         min_idx = x
        #
        # # append rec
        # normalized_config_vector = list_config_vector[min_idx]
        # return self.tf_config_util.config_vector_to_config_obj(
        #     self.tf_config_util.denormalize_config_vector(normalized_config_vector))
        shift = 0.001 # OtterTune GP will be fail if training data = sample data
        X_sample = []
        for x in self.normalized_training_data:
            x_without_loss = x[:-1]
            X_sample.append(x_without_loss+shift)
        X_sample = np.array(X_sample)
        current_loss = self.tf_config_util.l_scaler.transform(last_loss)[0][0]
        self.logger.debug("X_sample:%s " % str(X_sample))
        res = self.model.predict(X_sample, current_loss=current_loss)
        best_config_idx = np.argmin(res.minl.ravel())
        next_setting = res.minl_conf[best_config_idx, :]

        print "Raw gp ret:%s " % str(next_setting)

        next_setting_config_obj = self.tf_config_util.config_vector_to_config_obj(
            self.tf_config_util.denormalize_config_vector(next_setting),
            job_obj
        )

        logging.debug("Suggested new setting: %s" % str(next_setting_config_obj))

        return next_setting_config_obj


class TensorFlowConfigMetaData(object):
    def __init__(self, name, min_func, max_func, is_integer=True, is_categorical=False, default=0.0):
        self.name = name
        self.min_func = min_func
        self.max_func = max_func
        self.is_integer = is_integer
        self.is_categorical = is_categorical
        self.default = default

    def random(self):
        if self.is_integer:
            return random.randint(self.min_func(), self.max_func())
        else:
            return random.randrange(self.min_func(), self.max_func())


class TensorFlowConfigurationManager(object):
    def __init__(self, get_num_of_node_func, get_num_of_thread_func, learning_rate_range, batch_size_range):
        """
        :param FunctionType get_num_of_node_func:
        :param FunctionType get_num_of_thread_func:
        :param (float, float) learning_rate_range:
        :param (int, int) batch_size_range
        """
        self.logger = logging.getLogger(__name__)
        self.get_num_of_node_func = get_num_of_node_func
        self.get_num_of_thread_func = get_num_of_thread_func
        self.config_map = {}

        def ps_num_max_func():
            return get_num_of_node_func() - 1

        ps_num = TensorFlowConfigMetaData(_ps_num, min_func=lambda: 1, max_func=ps_num_max_func,
                                          is_integer=True, default=2)
        self.add_config_meta_data(ps_num)

        def thread_num_max_func():
            return get_num_of_thread_func() - 1

        intra_op_parallelism_threads = TensorFlowConfigMetaData(_intra_op_parallelism_threads, min_func=lambda: 1,
                                                                max_func=thread_num_max_func,
                                                                is_integer=True, default=8)
        self.add_config_meta_data(intra_op_parallelism_threads)

        inter_op_parallelism_threads = TensorFlowConfigMetaData(_inter_op_parallelism_threads, min_func=lambda: 1,
                                                                max_func=thread_num_max_func,
                                                                is_integer=True, default=8)
        self.add_config_meta_data(inter_op_parallelism_threads)

        # n_partition = TensorFlowConfigMetaData(_n_partition, min_func=lambda: 1,
        #                                         max_func=ps_num_max_func,
        #                                         is_discrete=True, default=1)
        # self.add_config_meta_data(n_partition)

        # learning_rate = TensorFlowConfigMetaData(_learning_rate, min_func=lambda: learning_rate_range[0], max_func=lambda: learning_rate_range[1],
        #                                          is_integer=False, default=0.01)
        # self.add_config_meta_data(learning_rate)
        #
        # batch_size = TensorFlowConfigMetaData(_batch_size, min_func=lambda: batch_size_range[0], max_func=lambda: batch_size_range[1], is_integer=True,
        #                                       default=100)
        # self.add_config_meta_data(batch_size)
        #
        # optimizer = TensorFlowConfigMetaData(_optimizer, min_func=lambda:0, max_func=lambda: len(
        #     selftf.lib.common_conf.optimizer_list) - 1,
        #                                      is_integer=True, default=1,
        #                                      is_categorical=True)
        # self.add_config_meta_data(optimizer)

        self.lhs_pending_config = {}

    def add_config_meta_data(self, meta_data):
        """
        :param TensorFlowConfigMetaData meta_data:
        :return: None
        """
        self.config_map[meta_data.name] = meta_data

    def get_tf_config_by_key_config(self, ps_num,
                                    intra_op_parallelism_threads,
                                    # n_partition,
                                    learning_rate,
                                    batch_size,
                                    optimizer):
        """
        :rtype: selftf.lib.common.PSTunerConfiguration
        """
        return PSTunerConfiguration(
            num_ps=ps_num,
            num_worker=self.get_num_of_node_func() - ps_num,
            intra_op_parallelism_threads=intra_op_parallelism_threads,
            inter_op_parallelism_threads=self.get_num_of_thread_func() - intra_op_parallelism_threads,
            # n_partition=n_partition,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer=optimizer
        )

    def get_tf_config_by_lhs(self, job_obj):
        """
        LHS generate enough sample for the beginning
        :param common.Job job_obj:
        :return:
        :rtype: PSTunerConfiguration
        """
        if self.lhs_pending_config.get(job_obj) is None:

            lhs_runner = LHSAdapter(self)
            self.lhs_pending_config[job_obj] = lhs_runner.get_batch_lhs_config(job_obj.os_size, job_obj)
            logging.debug("Generated LHS conf list: %s " % json.dumps(self.lhs_pending_config[job_obj],
                                                                      cls=common.PSTunerConfigurationSerializer))

        return self.lhs_pending_config[job_obj].pop()

    def get_tf_config_for_test_scheme1(self, job_obj):
        """
        LHS generate enough sample for the beginning
        :param common.Job job_obj:
        :return:
        :rtype: PSTunerConfiguration
        """
        if self.lhs_pending_config.get(job_obj) is None:
            lhs_runner = LHSAdapter(self)
            init_conf = lhs_runner.get_batch_lhs_config(job_obj.os_size, job_obj)[0]
            # modify intra thread only
            switch_conf = init_conf
            switch_conf.intra_op_parallelism_threads = switch_conf.intra_op_parallelism_threads + 1
            switch_conf.inter_op_parallelism_threads = switch_conf.inter_op_parallelism_threads - 1

            ret = []

            # fill up
            for x in range(0, job_obj.os_size):
                if x % 2 == 0:
                    ret.append(init_conf)
                else:
                    ret.append(switch_conf)

            self.lhs_pending_config[job_obj] = ret

        return self.lhs_pending_config[job_obj].pop()

    def get_tf_config_by_gp(self, job_obj):
        """
        :param  common.Job job_obj:
        :return:
        """

        online_sample_size = job_obj.get_online_os_size()
        training_data = job_obj.training_statistic.get()
        self.logger.info("Finish training phase, training GP mode with number of sample: %d" % online_sample_size)
        gp_model = GPTrainingModel(TFConfigUtil(self), epsilon=job_obj.get_target_loss(), func_name=job_obj.get_estimation_func_name())
        gp_model.train(training_data)

        conf = gp_model.get_best_config(sample_size=online_sample_size, list_existing_conf=job_obj.get_list_history_conf(),
                                        average_recovery_time=job_obj.get_avg_recovery_time(),
                                        last_loss=training_data[-1].loss,
                                        job_obj=job_obj)
        self.logger.info("Best training conf %s", json.dumps(conf.__dict__))
        return conf

    def get_tf_fixed_config(self, job_obj, total_num_of_node=0):
        """
                LHS generate enough sample for the beginning
                :param common.Job job_obj:
                :return:
                :rtype: PSTunerConfiguration
                """
        if self.lhs_pending_config.get(job_obj) is None:

            logging.debug("total_num_of_node:%d " % total_num_of_node)

            init_conf = PSTunerConfiguration(total_num_of_node - 1, 1, 14, 2, 1, 200, 3, 0.0001)
            switch_conf = PSTunerConfiguration(1, total_num_of_node - 1, 14, 2, 1, 200, 3, 0.0001)
            ret = []

            # fill up
            for x in range(0, job_obj.os_size):
                if x % 2 == 0:
                    ret.append(init_conf)
                else:
                    ret.append(switch_conf)

            self.lhs_pending_config[job_obj] = ret

        return self.lhs_pending_config[job_obj].pop()


class TFConfigUtil(object):
    def __init__(self, tf_config_manager):
        """
        :param TensorFlowConfigurationManager tf_config_manager:
        :return:
        """
        self.tf_config_manager = tf_config_manager
        self.tf_config_meta_data_map = tf_config_manager.config_map
        self.config_sequence = [
            _ps_num,
            _intra_op_parallelism_threads,
            # _n_partition,
            # _learning_rate,
            # _batch_size,
            # _optimizer
        ]
        self.range_matrix = self.get_range_matrix(self.tf_config_meta_data_map)

        categorical_encoder, x_scalar, y_scalar, constraint_helper = self.init_scalar_encoder(
            self.tf_config_meta_data_map, self.config_sequence
        )
        self.categorical_encoder = categorical_encoder
        self.constraint_helper = constraint_helper
        self.x_scalar = x_scalar
        self.y_scalar = y_scalar
        self.l_scaler = sklearn.preprocessing.StandardScaler()

        minX, maxX = self._get_min_max_vectors(
            self.tf_config_meta_data_map,
            self.config_sequence,
            self.categorical_encoder
        )

        self.denormalized_minX = minX
        self.denormalized_maxX = maxX

    def get_range_matrix(self, tf_config_meta_data_map):
        """
        :param dict[string, TensorFlowConfigMetaData] tf_config_meta_data_map:
        :return:
        """
        """
        	matrix: [n_ps, ps_strategy, sync_protocal, optimizer, batch size, learning_rate, individual op parallelism degree]
		n_ps: [[isInt, min , max]...]
		.....
		.....
		individual op parallelism degree: [[isInt, min , max]...]
        """
        # matrix = np.array([[1, 1, n_node],
        #                    [1, 1, 3],
        #                    [1, 1, 3],
        #                    [1, 1, 9],
        #                    [1, 1, 1000],
        #                    [0, 0.00001, 1],
        #                    [1, 0, n_threads]])
        ret_array = []
        for x in self.config_sequence:
            config_meta_data = tf_config_meta_data_map[x]
            ret_array.append([
                bool(config_meta_data.is_integer),
                config_meta_data.min_func(),
                config_meta_data.max_func()
            ])
        return ret_array

    def config_vector_to_config_obj(self, config_vector, job_obj):
        """
        :param common.Job job_obj:
        :param  list config_vector:
        :return:
        :rtype: PSTunerConfiguration
        """
        ps_num = config_vector[self.config_sequence.index(_ps_num)]
        intra_op_parallelism_threads = config_vector[self.config_sequence.index(_intra_op_parallelism_threads)]
        # n_partition = config_vector[self.config_sequence.index(_n_partition)]
        # learning_rate = config_vector[self.config_sequence.index(_learning_rate)]
        # batch_size = config_vector[self.config_sequence.index(_batch_size)]
        # optimizer= config_vector[self.config_sequence.index(_optimizer)]
        # return self.tf_config_manager.get_tf_config_by_key_config(ps_num=ps_num,
        #                                                           intra_op_parallelism_threads=intra_op_parallelism_threads,
        #                                                           # n_partition=n_partition,
        #                                                           learning_rate=learning_rate,
        #                                                           batch_size=batch_size,
        #                                                           optimizer=optimizer)
        return self.tf_config_manager.get_tf_config_by_key_config(ps_num=ps_num,
                                                                  intra_op_parallelism_threads=intra_op_parallelism_threads,
                                                                  # n_partition=n_partition,
                                                                  learning_rate=job_obj.learning_rate,
                                                                  batch_size=job_obj.batch_size,
                                                                  optimizer=job_obj.optimizer)

    def training_data_to_list_config_vector(self, list_tf_config_training_data):
        """
        :param list[PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """
        return map(lambda x: self.tf_config_to_vector(x.ps_config), list_tf_config_training_data)

    def tf_config_to_vector(self, tf_config):
        """
        :param PSTunerConfiguration tf_config:
        :return:
        """
        ret = []
        for x in self.config_sequence:
            ret.append(tf_config[x])
        return ret

    def list_tf_config_to_vector(self, list_tf_config):
        """
        :param list[PSTunerConfiguration] list_tf_config:
        :return:
        """
        ret = []
        for x in list_tf_config:
            ret.append(self.tf_config_to_vector(x))
        return ret

    def training_data_group_by_ps_config(self, list_tf_config_training_data):
        """
        :param list[PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """
        ret = {}
        for x in list_tf_config_training_data:
            if ret.get(x.ps_config, None) is None:
                ret[x.ps_config] = list()
            ret[x.ps_config].append(x)
        return ret

    @staticmethod
    def get_lastest_training_data_group_by_ps_config(list_tf_config_training_data):
        """
        Tested
        :param list[PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """
        ret = {}
        current_config = list_tf_config_training_data[0].ps_config
        for x in list_tf_config_training_data:
            if ret.get(x.ps_config, None) is None:
                ret[x.ps_config] = list()
            if x.ps_config != current_config: # new segment now, clear the old record
                ret[x.ps_config] = list()
            ret[x.ps_config].append(x)
            current_config = x.ps_config
        return ret

    def training_data_group_by_config_idx(self, list_tf_config_training_data):
        """
        :param list[PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """
        ret = {}
        for x in list_tf_config_training_data:
            if ret.get(x.ps_config_idx, None) is None:
                ret[x.ps_config_idx] = list()
            ret[x.ps_config_idx].append(x)
        return ret

    def list_training_data_get_average_time(self, list_tf_config_training_data):
        """
        :param list[PSTunerTrainingData] list_tf_config_training_data:
        :return:
        """
        sample_size = _average_runtime_sample
        if len(list_tf_config_training_data) < _average_runtime_sample:
            avg = self.cal_overall_time_per_iteration(list_tf_config_training_data)
        else:
            avg = self.cal_overall_time_per_iteration(
                list_tf_config_training_data[-sample_size:])
        logging.debug("list_training_data_get_average_time  avg:%f" % avg)
        return avg

    def cal_overall_time_per_iteration(self, list_training_data):
        """
        :param list[PSTunerTrainingData] list_training_data:
        :return:
        """
        start = list_training_data[0].timestamp - list_training_data[0].elapsed_time_in_ms
        end = list_training_data[-1].timestamp
        total_iteration = len(list_training_data)
        avg = (end-start)/ float(total_iteration)
        logging.debug("cal_overall_time_per_iteration start:%f, end:%f, avg:%f" % (start, end, avg))
        return avg

    # for LHS only
    def denormalize_config_vector_with_range_matrix(self, x):
        i = 0
        ret = []
        for item in x:
            dif = self.denormalize_value(item, self.range_matrix[i][1],
                                         self.range_matrix[i][2],
                                         self.range_matrix[i][0])
            ret.append(dif)
            i = i + 1
        return ret

    def denormalize_value(self, value, min, max, isInt):
        ret = value * (max - min ) + min
        if isInt:
            return int(ret)
        else:
            return ret

    def denormalize_config_vector(self, x):
        '''
        Args:
    	arr: an item of normalized test data
    	range_matrix: the range of each features
        Returns:
    	print the denormalized value of this item
        '''
        # i = 0
        # ret = []
        # for item in vector:
        #     dif = self.denormalize_value(item, self.range_matrix[i][1], self.range_matrix[i][2],
        #                                  self.range_matrix[i][0])
        #     ret.append(dif)
        #     i = i + 1
        # return ret
        X = np.array([x])
        X = self.x_scalar.inverse_transform(X)

        # For catagorical value
        if self.categorical_encoder is not None:
            X = self.categorical_encoder.inverse_transform(X)

        X = X[0]

        # convert categorical variables and int varaible to integer
        ret = []
        for idx, config_name in enumerate(self.config_sequence):
            metadata = self.tf_config_meta_data_map[config_name]
            if metadata.is_categorical or metadata.is_integer:
                min_value = metadata.min_func()
                max_value = metadata.max_func()
                value = int(X[idx])
                # check bound here
                if value < min_value:
                    value = min_value
                if value > max_value:
                    value = max_value

                ret.append(value)

            else:
                ret.append(X[idx])
        return ret

    def normalize_list_config_vector(self, list):
        X = np.array(list) # X matrix

        #categorical encode
        if self.categorical_encoder is not None:
            X = self.categorical_encoder.fit_transform(X)
        assert isinstance(X, np.ndarray)

        normalized_x_without_l = self.x_scalar.fit_transform(np.delete(X, X.shape[1]-1, 1))
        normalized_l = self.l_scaler.fit_transform((X[:,-1]).reshape(-1,1))

        return np.column_stack((normalized_x_without_l,normalized_l))


    def has_categorical_variable(self,tf_config_meta_data_map, config_sequence):
        """
        :param list[string] config_sequence:
        :param dict[string, TensorFlowConfigMetaData] tf_config_meta_data_map:
        """
        for var_name in config_sequence:
            meta_data = tf_config_meta_data_map[var_name]
            if meta_data.is_categorical:
                return True
        return False

    def init_scalar_encoder(self, tf_config_meta_data_map, config_sequence ):
        """
        :param list[string] config_sequence:
        :param dict[string, TensorFlowConfigMetaData] tf_config_meta_data_map:
        """

        x_scalar = sklearn.preprocessing.StandardScaler()
        y_scalar = sklearn.preprocessing.StandardScaler()

        # Check whether there are categorical variables
        if self.has_categorical_variable(tf_config_meta_data_map, config_sequence):
            categorical_encoder = self.init_categorical_encoder(tf_config_meta_data_map, config_sequence)
            constraint_helper = ParamConstraintHelper(scaler=x_scalar,
                                                      encoder=categorical_encoder)

        else:
            categorical_encoder = None
            constraint_helper = None

        return categorical_encoder, x_scalar, y_scalar, constraint_helper

    def init_categorical_encoder(self, tf_config_meta_data_map, config_sequence):
        """
        :param list[string] config_sequence:
        :param dict[string, TensorFlowConfigMetaData] tf_config_meta_data_map:
        """
        n_values = []
        cat_knob_indices = []
        cat_knob_names = []
        noncat_knob_names = []
        for idx, knob_name in enumerate(config_sequence):
            config_meta_data = tf_config_meta_data_map[knob_name]
            if config_meta_data.is_categorical:
                n_values.append((config_meta_data.max_func()-config_meta_data.min_func())+1)
                cat_knob_indices.append(idx)
                cat_knob_names.append(knob_name)
            else:
                noncat_knob_names.append(knob_name)
        return DummyEncoder(n_values,
                            cat_knob_indices,
                            cat_knob_names,
                            noncat_knob_names)

    def normalize_y(self, training_y):
        Y = np.vstack(training_y)
        return self.y_scalar.fit_transform(Y)

    def get_normalized_min_max_vectors(self):
        normalized_minX = self.x_scalar.transform(self.denormalized_minX.reshape(1,-1))[0]
        normalized_maxX = self.x_scalar.transform(self.denormalized_maxX.reshape(1,-1))[0]
        return normalized_minX, normalized_maxX

    @staticmethod
    def _get_min_max_vectors(tf_config_meta_data_map, config_sequence, categorical_encoder):
        """
        :param DummyEncoder categorical_encoder:
        :param list[string] config_sequence:
        :param dict[string, TensorFlowConfigMetaData] tf_config_meta_data_map:
        """
        min = []
        max = []
        if categorical_encoder is not None:
            for cat_idx in categorical_encoder.cat_idxs_old:
                metadata = tf_config_meta_data_map[config_sequence[cat_idx]]
                for x in range(metadata.min_func(), metadata.max_func() + 1):
                    min.append(0)
                    max.append(1)

        for idx, config_name in enumerate(config_sequence):
            metadata = tf_config_meta_data_map[config_sequence[idx]]
            if categorical_encoder is not None:
                if idx in categorical_encoder.cat_idxs_old:
                    continue
            min.append(metadata.min_func())
            max.append(metadata.max_func())
        return np.array(min), np.array(max)


class EstimationFuncUtil(object):

    # outlier removal func v3
    @staticmethod
    def outlier_median_discard(x_data, y_data, threshold=3):
        x_proc = x_data.copy()
        y_proc = y_data.copy()
        difference = np.abs(y_proc - np.median(y_proc))
        median_difference = np.median(difference)
        if median_difference == 0:
            s = np.zeros(y_proc.shape)
        else:
            s = difference / float(median_difference)
        mask = s < threshold
        y_proc = y_proc[mask]
        x_proc = x_proc[mask]
        return x_proc, y_proc

    @staticmethod
    def get_alpha_by_manual_least_square(list_x_loss, list_y_iteration, fit_func):
        xx = fit_func(list_x_loss, 1)
        yy = list_y_iteration
        alpha = np.true_divide(np.dot(xx, yy), np.dot(xx, xx))
        return alpha

    @staticmethod
    def get_alpha_by_old_bo_curve_fit(list_x_loss, list_y_iteration, fit_func):

        def fit_alpha(x_data, y_data, func):
            popt, pcov = curve_fit(func, x_data, y_data)
            return popt

        alpha = fit_alpha(list_x_loss, list_y_iteration, fit_func)
        return alpha

