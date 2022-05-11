import logging

import time

from selftf.lib import tf_program_util, common
import sys
import tensorflow as tf
import json

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                      datefmt='%m-%d %H:%M:%S',
                    )


class MLJobFramework:

    def print_trainable_variables(self, context):
        self.model_definition(context)
        print(json.dumps(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))

    def get_observe_loss_variable(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        return context.get_tf_variable_loss()

    def __init__(self):
        self.sess = None

    def model_definition(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        pass

    def get_feed_dict(self, context):
        """
        :param tf_program_util.TFProgramUtil context:
        :return:
        """
        return None

    def run(self):
        tftuner = tf_program_util.TFProgramUtil()

        # for collect statistic of trainable variable
        # should be executed by Monitor only
        if tftuner.is_collect_statistic_run():
            self.print_trainable_variables(tftuner)
            sys.exit(0)

        # cluster specification
        cluster = tftuner.get_tf_cluster_spec()

        server_config = tftuner.get_tf_config_proto()

        job_name = tftuner.get_job_name()
        task_index = tftuner.get_task_index()

        #set random seed
        tf.set_random_seed(task_index)

        server = tf.train.Server(
            cluster,
            job_name=job_name,
            task_index=task_index,
            config=server_config)

        tftuner.set_tf_server_target(server.target)
        tftuner.set_graph_init_func(self.model_definition)

        # Redirect all global variables (e.g global step) to the master ps
        with tftuner.get_default_tf_graph().as_default():
            with tf.device(tftuner.device_setter()):
                self.model_definition(tftuner)

                tftuner.init_graph()
                #
                # if tftuner.get_is_chief() and tftuner.conf_dict.get(common.conf_dict_non_static_ops_names) == None:
                #     # For first iteration get the static ops by chief
                #     # tftuner.set_chief_temp_static_ops(tftuner.get_static_ops())
                #     tftuner.set_chief_temp_nonstatic_ops(tftuner.get_non_static_ops())
                # tftuner.reallocate_static_ops()
                tf_program_util.SelfTFOptimizerContext.clear_all_ops_collocate()

        local_error = False

        try:
            while not tftuner.flag_end_process:
                tftuner.pre_recovery()
                if tftuner.is_ps():
                    if tftuner.do_live_reconfig2_context.is_phase_1():
                        tftuner.do_live_reconfig2_context.finish_phase1()
                    elif tftuner.do_live_reconfig2_context.is_final_phase():
                        tftuner.clear_do_scheme2_reconfig()

                    logging.debug("I am ps, I sleep")
                    time.sleep(1)

                    tftuner.post_check_reconfig_or_finish()

                else:

                    logging.info("Open new tensorflow session")
                    with tftuner.get_default_tf_graph().as_default():
                        with tftuner.get_monitored_training_session() as sess:
                            try:
                                logging.debug("Session is ready")
                                self.sess = sess

                                tftuner.post_recovery(sess=sess)
                                batch_time = time.time()
                                tftuner.pre_do_all_iteration(sess)
                                while True:

                                    feed_dict = self.get_feed_dict(tftuner)

                                    if feed_dict is None:
                                        _, cost, step = sess.run([tftuner.get_train_op(),
                                                                  self.get_observe_loss_variable(tftuner),
                                                                  tftuner.get_tf_variable_global_step()])
                                    else:
                                        _, cost, step = sess.run(
                                            [tftuner.get_train_op(),
                                             self.get_observe_loss_variable(tftuner),
                                             tftuner.get_tf_variable_global_step()],
                                            feed_dict=feed_dict)
                                    tftuner.post_do_iteration(steps=step, loss=cost, timestamp=time.time(),
                                                              duration=time.time() - batch_time)
                                    batch_time = time.time()

                                    if (tftuner.should_stop_iteration(step, cost)
                                        or tftuner.is_reconfig()
                                        or tftuner.is_reconfig2()
                                        or tftuner.flag_do_checkpoint
                                        or tftuner.flag_end_process):
                                        break

                            except (KeyboardInterrupt, SystemExit):
                                logging.exception("System should be exited herfe")
                                pass
                            except:
                                logging.exception("Something Wrong")
                                local_error = True # Turn this flag on the restart it
                            finally:
                                if not local_error:
                                    # Regular program exit
                                    tftuner.post_do_all_iteration(sess)
                    tftuner.post_check_reconfig_or_finish()
        except:
            logging.exception("Error")
        finally:
            sys.exit(0)


