import tensorflow as tf
import dataloader
dataset = dataloader.load_dataSets()
from TF_3D_ops import BatchNormalization, Conv3D, Concatenate, MaxPooling3D,\
            Deconv3D
from config import FLAGS
import sys
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES']='0'


volume_input = tf.placeholder(shape=(FLAGS.batch_size,32,32,32,1), 
                    name='mainInput', dtype=tf.float32)
segmentation_map = tf.placeholder(shape=(FLAGS.batch_size, 32,32,32), 
                    name="segmentation_map", dtype=tf.float32)

class DenseVoxNet(object):
    """docstring for DenseNet"""

    def denseBlock(self, layers, name, volumes_):
        for i in xrange(layers):
            BN = BatchNormalization(name='bn_'+name+str(i),
                        data=volumes_,
                        training=FLAGS.train_mode)
            conv = Conv3D(filters=32, 
                    name='convolution_'+name+str(i),
                    kernel_size=[3,3,3],
                    activation=None,
                    data=BN)
            volumes_ = Concatenate([volumes_, conv], 
                    name='concatenate_'+name+str(i))
        return volumes_

    def graph(self, volume_input):
        layer1 = Conv3D(filters=16, 
                name='layer1', 
                kernel_size=[3,3,3],
                # activation=None, 
                data=volume_input)

        for i in range(FLAGS.stages):
            if i == 0: data = layer1
            else: data = maxpool
            
            dense = self.denseBlock(12,"Down_denseBlock_layer"+str(i),data)

            BN = BatchNormalization(name="Down_BN_layer"+str(i), 
                    data=dense, 
                    training=FLAGS.train_mode)
            conv = Conv3D(filters=30*(i+1), 
                    name='Down_conv_layer'+str(i), 
                    kernel_size=(1,1,1),
                    activation=None,
                    data=BN)
            maxpool = MaxPooling3D(data=conv, 
                    pool_size=(2,2,2), 
                    name='Down_pool_layer'+str(i))
            print maxpool.get_shape()

        B_Bnorm = BatchNormalization(name="BN_Batch_Norm", 
                data=maxpool, 
                training=FLAGS.train_mode)
        B_Conv = Conv3D(filters=304, 
                name='BN_Conv', 
                kernel_size=(1,1,1),
                activation=None,
                data=B_Bnorm)

        for i in range(FLAGS.stages):
            if i == 0: data = B_Conv
            else: data = dense 

            deconv = Deconv3D(data=data, 
                    filters=30*(4-i), 
                    kernel_size=(3,3,3),
                    strides=(2,2,2),
                    name="deconv_"+str(i))

            BN = BatchNormalization(name="Up_BN_layer"+str(i), 
                    data=deconv, 
                    training=FLAGS.train_mode)
            print BN.get_shape()
            dense = self.denseBlock(12,"Up_denseBlock_"+str(i),BN)

        output = Conv3D(data=dense, 
                    filters=1, 
                    kernel_size=(3,3,3),
                    name="conv_output")
	output = tf.reshape(output, shape=(FLAGS.batch_size, 32,32,32))
        return output

    def loss(self, pred, segmentation_map):
        error = tf.contrib.losses.sigmoid_cross_entropy(
                pred,
                multi_class_labels=segmentation_map,
                weights=1.0,
                scope="error")
        tf.summary.scalar("cost", error)
        return error

    def optimizer(self, cost, global_step):
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                global_step,
                FLAGS.decay_steps,
                FLAGS.decay_factor,
                staircase=True)
        tf.summary.scalar("lr", lr)
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)



def train():
    # param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #       tf.get_default_graph(),
    #       tfprof_options=tf.contrib.tfprof.model_analyzer.
    #               TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    #sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    network = DenseVoxNet()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(1), trainable=False)
            
        # with  tf.device(FLAGS.device):
        step = 0
        pred = network.graph(volume_input)
        cost = network.loss(pred, segmentation_map)
        opti = network.optimizer(cost, global_step)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        while dataset.train.epochs_completed <= FLAGS.epochs:
            volume, seg_map = dataset.train.next_batch(FLAGS.batch_size)
            print list(seg_map[0][0])
            _, loss, summary = sess.run([opti, cost, merged], feed_dict = { volume_input:volume, segmentation_map: seg_map})
            train_writer.add_summary(summary, step)
            
            if step %  FLAGS.eval_interval == 0:
                test_volume, test_seg_map = dataset.test.volumes, dataset.test.segmentations

                _, loss, summary = sess.run([opti, cost], 
                    feed_dict = { volume_input:test_volume, segmentation_map: test_seg_map})

            if step % FLAGS.save_model == 0:
                save_path = saver.save(sess, FLAGS.checkpoint_dir)

            if step % FLAGS.log_frequency ==0:
                print "step :{}, epochs: {}, loss: {}".format(step, dataset.epochs_completed, loss)
            step +=1
    pass

if __name__ == "__main__":
    train()