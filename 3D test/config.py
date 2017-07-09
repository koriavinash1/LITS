import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', "../../DataSet/NewDataSet/train",
		"""Directory to load train data""")
tf.app.flags.DEFINE_string('checkpoint_dir', '../../results/logs/model.ckpt',
		"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('log_dir','../../results/logs/model.ckpt',
		"""Directory to save logs and models""")

tf.app.flags.DEFINE_integer('epochs', 100 ,
		"""Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
		"""How often to log results to the console.""")
tf.app.flags.DEFINE_float('dropout',0.25,
		"""Keep Probability""")
tf.app.flags.DEFINE_integer('batch_size',1,
		"""Batch size for network input.""")
tf.app.flags.DEFINE_boolean('train_mode',True,
		"""Mode to use BatchNormalization""")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
		"""Initial Learning Rate""")
tf.app.flags.DEFINE_integer('stages', 2,
		"""Number of DenseBlocks in up & down sampling.""")

tf.app.flags.DEFINE_integer('eval_interval', 1,
		"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('save_model', 5,
		"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('decay_steps',10,
		"""Display on screen""")
tf.app.flags.DEFINE_float('decay_factor',0.2,
		"""Learning rate decay rate""")

tf.app.flags.DEFINE_string('device', '/gpu:0',
		"""Device for training""")
