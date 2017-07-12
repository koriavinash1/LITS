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


volume_input = tf.placeholder(shape=(FLAGS.batch_size,128,128,128,1), 
                    name='mainInput', dtype=tf.float32)
segmentation_map = tf.placeholder(shape=(FLAGS.batch_size, 128,128,128), 
                    name="segmentation_map", dtype=tf.float32)

class DenseVoxNet(object):
    """docstring for DenseNet"""

    def denseBlock(self, layers, name, volumes_):
        for i in xrange(layers):
            concat_layers = []
            concat_layers.append(volumes_)
            BN = BatchNormalization(name='bn_'+name+str(i),
                        data=volumes_,
                        training=FLAGS.train_mode)
            conv = Conv3D(filters=32, 
                    name='convolution_'+name+str(i),
                    kernel_size=[3,3,3],
                    activation=None,
                    data=BN)
            for j in range(len(concat_layers)-1):
                volumes_ = concat_layers[j]
                volumes_ = Concatenate([volumes_, concat_layers[i]], 
                        name='concatenate_'+name+str(j)+str(i))
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
            
            dense = self.denseBlock(FLAGS.denseBlockDepth,"Down_denseBlock_layer"+str(i),data)

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
            print maxpool

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
                    filters=30*(FLAGS.stages-i), 
                    kernel_size=(3,3,3),
                    strides=(2,2,2),
                    name="deconv_"+str(i))

            BN = BatchNormalization(name="Up_BN_layer"+str(i), 
                    data=deconv, 
                    training=FLAGS.train_mode)
            print BN, i
            dense = self.denseBlock(FLAGS.denseBlockDepth,"Up_denseBlock_"+str(i),BN)

        output = Conv3D(data=dense, 
                    filters=1, 
                    kernel_size=(3,3,3),
                    name="conv_output")
        output = tf.reshape(output, shape=(FLAGS.batch_size, 128,128,128))
        print output
        return output

    def status(self):
        print "Down and Up sampling stages: {}, DenseBlockDepth: {},\
                    BatchSize: {}, Epochs: {}, LogFreq: {}, ModelSavingFreq: {}"\
                    .format(FLAGS.stages, FLAGS.denseBlockDepth,\
                        FLAGS.batch_size, FLAGS.epochs, FLAGS.log_frequency,
                        FLAGS.save_model)
        # print "##########################################"
        pass

    def count_variables(self):    
        total_parameters = 0
        #iterating over all variables
        for variable in tf.trainable_variables():  
            local_parameters=1
            shape = variable.get_shape()  #getting shape of a variable
            for i in shape:
                local_parameters*=i.value  #mutiplying dimension values
            total_parameters+=local_parameters
        print('Total Number of Trainable Parameters:', total_parameters) 

    def loss(self, pred, segmentation_map):
        error = tf.contrib.losses.softmax_cross_entropy(
                logits = pred,
                onehot_labels=segmentation_map,
                weights=1.0,
                label_smoothing=0,
                scope="loss")
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

    def IOU_cal(self, pred_val, true_val):
        iou, conf_matrix = tf.metrics.mean_iou(true_val, 
                                tf.argmax(pred_val, 3), 3,
                                name = "IOU_measure")
        tf.summary.scalar("iou", iou)
        return iou, conf_matrix

    def diceScore(self, yPred, yTruth, thresh):
        smooth = tf.constant(1.0)
        mul = tf.argmax(yPred, 3)*yTruth
        intersection = 2*tf.reduce_sum(mul) + smooth
        union = tf.reduce_sum(yPred) + tf.reduce_sum(yTruth) + smooth
        dice = intersection/union
        return dice

    def define_full_model(self, 
                                        volume_input, 
                                        segmentation_map):
        global_step = tf.get_variable('global_step', [],
                initializer=tf.constant_initializer(1), trainable=False)
        predictions = self.graph(volume_input)
        cost = self.loss(predictions, segmentation_map)
        optimizer_value = self.optimizer(cost, global_step)
        iou, conf_matrix = self.IOU_cal(predictions, segmentation_map)
        dice = self.diceScore(volume_input, segmentation_mapZ)
        return global_step, optimizer_value, iou, cost, dice


def train():
    # param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #       tf.get_default_graph(),
    #       tfprof_options=tf.contrib.tfprof.model_analyzer.
    #               TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    #sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
    pid = os.getpid()
    network = DenseVoxNet()
    network.status()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    global_step, opti, iou, cost, dice = network.define_full_model(volume_input, segmentation_map)

    with tf.Session(config=config) as sess:
	
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        network.count_variables()
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            
        # with  tf.device(FLAGS.device):
        step = 0
        while dataset.train.epochs_completed <= FLAGS.epochs:
            volume, seg_map = dataset.train.next_batch(FLAGS.batch_size)

            _, loss,iou_info,summary, score = sess.run([opti, cost, iou, merged, dice], feed_dict = { volume_input:volume, segmentation_map: seg_map})
            train_writer.add_summary(summary, step)
            
            if step %  FLAGS.eval_interval == 0:
                test_volume, test_seg_map = dataset.test.volumes, dataset.test.segmentations
                test_volume = np.reshape(test_volume, (FLAGS.batch_size,128,128,128,1))
                _, loss, iou_info = sess.run([opti, iou, cost], feed_dict = { volume_input:test_volume, segmentation_map: test_seg_map})

            if step % FLAGS.save_model == 0:
                save_path = saver.save(sess, FLAGS.checkpoint_dir)

            if step % FLAGS.log_frequency ==0:
                print "[{}] step :{}, IOU %: {}, epochs: {}, loss: {}, diceScore: {}".format(pid, step, iou_info*100.0, dataset.train.epochs_completed, loss, score)
            step +=1
    pass

if __name__ == "__main__":
    train()