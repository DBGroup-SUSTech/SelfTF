import tensorflow as tf
from selftf.lib.ml_job_framework import MLJobFramework
import selftf.tf_job.inception.inception_model as inception
from selftf.tf_job.inception.slim import slim
import selftf.tf_job.inception.image_processing as image_processing
from selftf.tf_job.inception.imagenet_data import ImagenetData

num_preprocess_threads = 4
dataset = ImagenetData(subset="train")


class InceptionV3(MLJobFramework):

    def model_definition(self, context):
        # Variables and its related init/assign ops are assigned to ps.
        # with slim.scopes.arg_scope(
        #     [slim.variables.variable, slim.variables.global_step],
        #     device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
      # Create a variable to count the number of train() calls. This equals the
      # number of updates applied to the variables.
      global_step = slim.variables.global_step()
      context.set_global_step(global_step)

      # Calculate the learning rate schedule.
      # num_batches_per_epoch = (dataset.num_examples_per_epoch() /
      #                          FLAGS.batch_size)
      # Decay steps need to be divided by the number of replicas to aggregate.
      # decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay /
      #                   num_replicas_to_aggregate)

      # Decay the learning rate exponentially based on the number of steps.
      # lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
      #                                 global_step,
      #                                 decay_steps,
      #                                 FLAGS.learning_rate_decay_factor,
      #                                 staircase=True)
      # Add a summary to track the learning rate.
      # tf.summary.scalar('learning_rate', lr)

      # # Create an optimizer that performs gradient descent.
      # opt = tf.train.RMSPropOptimizer(lr,
      #                                 RMSPROP_DECAY,`
      #                                 momentum=RMSPROP_MOMENTUM,
      #                                 epsilon=RMSPROP_EPSILON)
      images, labels = image_processing.distorted_inputs(
          dataset,
          batch_size=context.get_batch_size(),
          num_preprocess_threads=num_preprocess_threads)

      # Number of classes in the Dataset label set plus 1.
      # Label 0 is reserved for an (unused) background class.
      num_classes = dataset.num_classes() + 1
      logits = inception.inference(images, num_classes, for_training=True)
      # Add classification loss.
      inception.loss(logits, labels, batch_size=context.get_batch_size())

      # Gather all of the losses including regularization losses.
      losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
      losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

      total_loss = tf.add_n(losses, name='total_loss')

      context.set_train_op(total_loss)

if __name__ == "__main__":
    model = InceptionV3()
    model.run()
