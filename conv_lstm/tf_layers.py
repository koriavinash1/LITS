import numpy as np
import tensorflow as tf
import os
import sys
from datetime import datetime
import time
sys.path.append("../")
# Custom
from data_preprocess import utils
from tf_ops import conv2d, BN_ReLU_Conv, TransitionDown, TransitionUp, FinalLayer, conv_rnn 
from data_loader.loader import load_data, load_data_full_test, extract_patch, pad_patch

class DFCN(object):
    def __init__(self,
                 sess,
                 conf,  
                 # Architecture Specific
                 # 16 -> 3*3 : 4-> 5*5 : 4 -> 7*7
                 # total =24
                 n_filters_first_conv=[16, 4, 4],
                 # 3, 4, 5, 8, 5, 4, 3
                 n_layers_per_block=[3, 4, 5, 8, 5, 4, 3]):
        """
        Tensorflow implementation of modified version of Tiramasu
        Source code inspiration: 
        https://github.com/SimJeg/FC-DenseNet
        https://github.com/divelab/dtn 

        This code implements the Fully Convolutional DenseNet described in https://arxiv.org/abs/1611.09326
        The network consist of a downsampling path, where dense blocks and transition down are applied, followed
        by an upsampling path where transition up and dense blocks are applied.
        Skip connections are used between the downsampling path and the upsampling path
        Each layer is a composite function of BN - ReLU - Conv and the last layer is a softmax layer.

        :param n_filters_first_conv: number of filters for the first convolution applied
        :param n_pool: number of pooling layers = number of transition down = number of transition up
        :param growth_rate: number of new feature maps created by each layer in a dense block
        :param n_layers_per_block: number of layers per block. Can be an int or a list of size 2 * n_pool + 1
        """

        if type(n_layers_per_block) == list:
            assert (len(n_layers_per_block) == 2 * conf.n_pool + 1)
        elif type(n_layers_per_block) == int:
            n_layers_per_block = [n_layers_per_block] * (2 * conf.n_pool + 1)
        else:
            raise ValueError

        self.sess = sess
        self.conf = conf
        self.input_shape = self.input_shape = [conf.batch_size,
                                 conf.height, conf.width, conf.channels]
        self.output_shape = [conf.batch_size, conf.height, conf.width]                                 
        self.channel_axis = 3
        self.n_filters_first_conv = n_filters_first_conv 
        self.n_layers_per_block = n_layers_per_block                             
        self.n_pool = conf.n_pool
        self.growth_rate = conf.growth_rate
        self.dropout_p = conf.dropout_p
        self.weight_decay = conf.weight_decay

        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sample_dir):
            os.makedirs(conf.sample_dir)
        if not os.path.exists(conf.best_model_dir):
            os.makedirs(conf.best_model_dir)

        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        # Lambda Function: For metric 
        init_history = lambda: {'loss': [], 'dice': [], 'jaccard': [], 'accuracy': []}
        self.history = {'train': init_history(), 'val': init_history(), 'test': init_history()}
        self.patience = conf.patience
        self.best_epoch = 0
        self.best_jaccard_score = 0.
        self.best_dice = 0.
        self.best_accuracy = 0.

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        summarys.append(tf.summary.image(
            name+'/input', self.inputs, max_outputs=10))
        summarys.append(tf.summary.image(
            name +
            '/annotation', tf.cast(tf.expand_dims(
                self.annotations, -1), tf.float32),
            max_outputs=10))
        summarys.append(tf.summary.image(
            name +
            '/prediction', tf.cast(tf.expand_dims(
                self.decoded_predictions, -1), tf.float32),
            max_outputs=10))
        summary = tf.summary.merge(summarys)
        return summary

    def featuremap_summary(self, feature_map_list, name):
        summarys = []
        for cnt, fmap in enumerate(feature_map_list):
            summarys.append(tf.summary.image(name+'/fmap%2'%cnt, fmap, max_outputs=25))
            summary = tf.summary.merge(summarys)
        return summary

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

    def configure_networks(self):
        self.build_network()
        optimizer = tf.train.AdamOptimizer(self.conf.learning_rate)
        # Batch Norm: Related dependencies
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.count_variables()

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.annotations = tf.placeholder(
            tf.int64, self.output_shape, name='annotations')
        self.is_training = tf.placeholder(tf.bool, shape=[],
                 name='batch_norm_control')
        
        expand_annotations = tf.expand_dims(
            self.annotations, -1, name='annotations/expand_dims')
        one_hot_annotations = tf.squeeze(
            expand_annotations, axis=[self.channel_axis],
            name='annotations/squeeze')
        one_hot_annotations = tf.one_hot(
            one_hot_annotations, depth=self.conf.n_classes,
            axis=self.channel_axis, name='annotations/one_hot')

        # Prediction
        self.predictions = self.inference(self.inputs)
        losses = tf.losses.softmax_cross_entropy(
            one_hot_annotations, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        self.decoded_predictions = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.annotations, self.decoded_predictions,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        # TODO:If tf.greater_equal is implemented then background label->0 is
        # not considered
        weights = tf.cast(
            tf.greater(self.decoded_predictions, 0, name='m_iou/greater'),
            tf.int32, name='m_iou/weights')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(
            self.annotations, self.decoded_predictions, self.conf.n_classes,
            weights, name='m_iou/m_ious')

    def construct_mini_inception(self, inputs, name='mini_inception', 
                                activation_fn=tf.nn.relu):
        """
        Build the initial block on filters
        """
        stack = []
        l = conv2d(inputs, self.n_filters_first_conv[0],
            kernel_size=3, name=name+'/3_3', activation_fn=activation_fn)
        stack.append(l)
        l = conv2d(inputs, self.n_filters_first_conv[1],
            kernel_size=5, name=name+'/5_5', activation_fn=activation_fn)       
        stack.append(l)
        l = conv2d(inputs, self.n_filters_first_conv[2],
            kernel_size=7, name=name+'/7_7', activation_fn=activation_fn)
        stack.append(l) 
        # The number of feature maps in the stack is stored in the variable n_filters
        stack = tf.concat(stack, self.channel_axis, name=name+'/concat')
        return stack 

    def construct_DFCN(self, inputs):
        """
        Implementation of modified version of Tiramisu
        """
        skip_connection_list = []
        ###############################################
        # First mini inception type convolution layer #
        ###############################################
        stack = self.construct_mini_inception(inputs)
     
        print("First Layer shape ", stack.get_shape().as_list())   
        n_filters = sum(self.n_filters_first_conv)
        for i in range(self.n_pool):
            # Dense Block
            name = 'DB_Down%s' % i
            for j in range(self.n_layers_per_block[i]):
                # Compute new feature maps
                l = BN_ReLU_Conv(stack, self.growth_rate, 
                    self.is_training, name=name+'/layer%s'%j,
                    dropout_p=self.dropout_p)
                # And stack it : the Tiramisu is growing
                stack = tf.concat([stack, l], self.channel_axis, name=name+'/concat')
                n_filters += self.growth_rate
            print("DB_Down:", i, " shape ", stack.get_shape().as_list()) 
            # At the end of the dense block, the current stack is stored
            # in the skip_connections list
            skip_connection_list.append(stack)

            # Transition Down
            name = 'TD%s' % i
            stack = TransitionDown(stack, n_filters, self.is_training, 
                    name, dropout_p=self.dropout_p)
            print("TD:", i, " shape ", stack.get_shape().as_list())  
            # if size is reduced by half
        skip_connection_list = skip_connection_list[::-1]
        #####################
        #     Bottleneck    #
        #####################

        # We store now the output of the next dense block in a list.
        # We will only upsample these new feature maps
        block_to_upsample = []

        # Dense Block
        name = 'DB_Bottleneck'
        for j in range(self.n_layers_per_block[self.n_pool]):
            l = BN_ReLU_Conv(stack, self.growth_rate, 
                self.is_training, name=name+'/layer%s'%j,
                dropout_p=self.dropout_p)
            block_to_upsample.append(l)
            stack = tf.concat([stack, l], self.channel_axis, name=name+'/concat')
        print("DB_Bottleneck:", " shape ", stack.get_shape().as_list())

        #############################
        #  Convolution LSTM/GRU   # 
        ############################# 
        # Append these lists to inputs of sess.run after every batch and every new patient respectively 
        self.state_update_ops_list = []
        self.reset_ops_list = []
        for i in range(self.n_pool):
            skip_connection_list[i], state_update_ops, reset_ops = conv_rnn(skip_connection_list[i], cell_type='LSTM', name='conv_lstm%d'%i)
            self.state_update_ops_list.append(state_update_ops)
            self.reset_ops_list.append(reset_ops)

        #######################
        #   Upsampling path   #
        #######################                                            
        for i in range(self.n_pool):
            name = 'TU%s' % i
            # Transition Up ( Upsampling + concatenation with the skip connection)
            n_filters_keep = self.growth_rate * self.n_layers_per_block[self.n_pool + i]
            stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep,
                        name, self.channel_axis)
            print("TU:", i, " shape ", stack.get_shape().as_list())  
            # Dense Block
            block_to_upsample = []
            name = 'DB_Up%s' % i
            for j in range(self.n_layers_per_block[self.n_pool + i + 1]):
                l = BN_ReLU_Conv(stack, self.growth_rate, 
                    self.is_training, name=name+'/layer%s'%j,
                    dropout_p=self.dropout_p)
                block_to_upsample.append(l)
                stack = tf.concat([stack, l], self.channel_axis, name=name+'/concat')
            print("DB_Up:", i, " shape ", stack.get_shape().as_list())
        output = FinalLayer(stack, self.conf.n_classes, name='Final_Layer')

        print("Final Layer:", " shape ", output.get_shape().as_list())
        return output

    def inference(self, inputs):
        """
        Build the network: DFCN
        """
        outputs = self.construct_DFCN(inputs)
        return outputs

    def save_summary(self, summary, epoch_num):
        #print('---->summarizing', epoch_num)
        self.writer.add_summary(summary, epoch_num)

    def save(self, epoch_num):
        print('---->saving', epoch_num)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch_num)

    def reload(self, epoch_num):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(epoch_num)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def save_best_model(self):
        checkpoint_path = os.path.join(
            self.conf.best_model_dir, 'best_model')
        self.saver.save(self.sess, checkpoint_path, global_step=None)

    def load_best_model(self):
        checkpoint_path = os.path.join(
            self.conf.best_model_dir, 'best_model')
        model_path = checkpoint_path
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def batch_loop(self, iterator, epoch, phase, history):
        """ Loop on the batches """
        n_batches = iterator.get_n_batches()
        self.sess.run(tf.local_variables_initializer())
        predictions = []
        losses = []
        accuracies = []
        m_ious = []
        n_imgs = 0.
        count=0
        prev_pid=None
        for inputs, annotations, pid in iterator():
            # print pid, prev_pid
            if prev_pid != pid[0]:
                self.sess.run(self.reset_ops_list)
                prev_pid = pid[0]
            
            feed_dict = {self.inputs: inputs,
                         self.annotations: annotations,
                         self.is_training: 1 if phase == 'train' else 0}
            if phase == 'train':
                loss, summary, accuracy, m_iou = self.sess.run(
                    [self.loss_op, self.train_summary,
                    self.accuracy_op, self.m_iou, self.train_op, self.miou_op]+self.state_update_ops_list,
                    feed_dict=feed_dict)[0:4]
                self.save_summary(summary, epoch)
            elif phase == 'val':
                loss, summary, accuracy, predictions, m_iou = self.sess.run(
                    [self.loss_op, self.valid_summary, self.accuracy_op,
                    self.decoded_predictions, 
                    self.m_iou, self.miou_op]+self.state_update_ops_list, feed_dict=feed_dict)[0:5]             
                self.save_summary(summary, epoch)      
            else:
                loss, accuracy, predictions, m_iou = self.sess.run(
                [self.loss_op, self.accuracy_op,
                self.decoded_predictions, 
                self.m_iou, self.miou_op]+self.state_update_ops_list, feed_dict=feed_dict)[0:4]   
            # print('values----->', loss, accuracy, m_iou)
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
            count+=1
            # # Progression bar ( < 74 characters)
            sys.stdout.write('\rEpoch {} : [{} : {}%]'.format(epoch, phase, int(100. * (count + 1) / n_batches)))
            sys.stdout.flush()
        self.history[phase]['loss'].append(np.mean(losses))
        self.history[phase]['dice'].append(0)
        self.history[phase]['jaccard'].append(m_ious[-1])
        self.history[phase]['accuracy'].append(np.mean(accuracies)) 

    def train(self):
        ###############
        #  load data  #
        ###############

        print('-' * 75)
        print('Loading data')

        train_iter, val_iter, test_iter = load_data(self.conf.max_batch_size if not self.conf.batch_size
                                                     else self.conf.batch_size,
                                             self.conf.patient_batch_size)
        print('Number of Patients : train : {}, val : {}, test : {}'.format(
        train_iter.get_n_samples(), val_iter.get_n_samples(), test_iter.get_n_samples()))
        
        # Initialize the generator
        if self.conf.num_epochs==0:
            # Load best model weights
            self.load_best_model()
            # Test
            print('Training ends\nTest')
            if test_iter.get_n_samples() == 0:
                print 'No test set'
            else:
                self.batch_loop(val_iter, 0, 'test', self.history)

                print ('Average cost test = {:.5f} | jacc test = {:.5f} | acc_test = {:.5f} | dice_test = {:.5f}'.format(
                    self.history['test']['loss'][-1],
                    self.history['test']['jaccard'][-1],
                    self.history['test']['accuracy'][-1],
                    self.history['test']['dice'][-1]
                    ))
            # Exit
            return

        # Reload from a certain epoch
        if self.conf.reload_epoch > 0:
            print ('***************Trying to reload model******************')
            self.reload(self.conf.reload_epoch)
        # Training main loop
        print('-' * 30)
        print('Training starts at ' + str(datetime.now()).split('.')[0])
        print('-' * 30)

        for epoch_num in range(self.conf.num_epochs):
            # Train
            start_time_train = time.time()
            # TODO:
            self.batch_loop(train_iter, epoch_num, 'train', self.history)
            if epoch_num % self.conf.save_step == 1:
                self.save(epoch_num)
            # Validation
            start_time_valid = time.time()
            self.batch_loop(val_iter, epoch_num, 'val', self.history)
                # Print
            out_str = \
                '\r\x1b[Epoch {} took {}+{} sec. ' \
                'loss = {:.5f} | dice = {:.5f} | jacc = {:.5f} | acc = {:.5f} || ' \
                'loss = {:.5f} | dice = {:.5f} | jacc = {:.5f} | acc = {:.5f}'.format(
                    epoch_num, int(start_time_valid - start_time_train), int(time.time() - start_time_valid),
                    self.history['train']['loss'][-1], self.history['train']['dice'][-1], self.history['train']['jaccard'][-1], self.history['train']['accuracy'][-1],
                    self.history['val']['loss'][-1], self.history['val']['dice'][-1], self.history['val']['jaccard'][-1], self.history['val']['accuracy'][-1])

            np.savez(os.path.join( self.conf.best_model_dir, 'history.npz'), metrics=self.history,
                     best_epoch=self.best_epoch)    

            # Monitoring jaccard:
            if self.history['val']['jaccard'][-1] > self.best_jaccard_score:
                out_str += ' (BEST jaccard)'
                self.best_jaccard_score = self.history['val']['jaccard'][-1]
                self.best_epoch = epoch_num
                self.patience = 0
                self.save_best_model()
            else:
                self.patience += 1
            print (out_str)

            # Finish training if patience has expired or max no. of epochs reached
            if self.patience == self.conf.patience or epoch_num == self.conf.num_epochs - 1:
                # Load best model weights
                self.load_best_model()
                # Test
                print('Training ends\nTest')
                if test_iter.get_n_samples() == 0:
                    print 'No test set'
                else:
                    self.batch_loop(test_iter, epoch_num, 'test', self.history)

                    print ('Average cost test = {:.5f} | jacc test = {:.5f} | acc_test = {:.5f} | dice_test = {:.5f}'.format(
                        self.history['test']['loss'][-1],
                        self.history['test']['jaccard'][-1],
                        self.history['test']['accuracy'][-1],
                        self.history['test']['dice'][-1]
                        ))
                return

    def predict_feeder(self, X):
        """
        Takes Input image and GT gives back prediction
        """
        feed_dict = {self.inputs: X, self.is_training:False}
        predictions = self.sess.run(
            [self.decoded_predictions],
            feed_dict=feed_dict)
        return predictions[0]

    def predict_slicewise(self, X):
        n_slices, nrows, ncols,_ = X.shape
        volume = np.empty((0, nrows, ncols), dtype=np.uint8)
        for i in range(n_slices):
            feed_dict = {self.inputs: np.expand_dims(X[i], axis=0), self.is_training:False}
            _slice = self.sess.run(
                        [self.decoded_predictions],
                        feed_dict=feed_dict)
            # print _slice[0].shape
            volume = np.append(volume,_slice[0], axis=0)
        return volume

    def predict(self, path):
        print('---->predicting ', self.conf.test_epoch)
        if self.conf.test_epoch > 0:
            print ('Reloading Model from epoch No', self.conf.test_epoch)
            self.reload(self.conf.test_epoch)
        else:
            print("Reloading Best Model based on Jaccard Score")
            # Load best model weights
            self.load_best_model()

        self.sess.run(tf.local_variables_initializer())
        approach=2
        # If patientwise ED and ES prediction
        # Load data
        print('Loading data')
        iterator = load_data_full_test()
        _iter = iterator.test_generator()
        n_batches = iterator.get_n_patient_batches()
        for i in range(n_batches):
            X, Y, pid, affine, hdr, roi_mask, roi_center  = _iter.next()
            batch_size = X.shape[0]
            n_slices = batch_size/2
            ed_img = X[:n_slices]
            ed_gt = Y[:n_slices]
            es_img = X[n_slices:]
            es_gt = Y[n_slices:]    
            try:            
                print pid
                if approach==1:
                    ed_patch_img, pad_params = extract_patch(ed_img, roi_center, patch_size=128)
                    es_patch_img, _ = extract_patch(es_img, roi_center, patch_size=128)           

                    ed_pred_gt = self.predict_slicewise(ed_patch_img)
                    print('Patient-ED', pid)
                    es_pred_gt = self.predict_slicewise(es_patch_img)
                    print('Patient-ES', pid)
                    # pad the predictions and translate
                    ed_pred_gt_pad = pad_patch(ed_pred_gt, pad_params)
                    es_pred_gt_pad = pad_patch(es_pred_gt, pad_params)
                    # Post processing mask 
                    ed_pred_gt_pad *= roi_mask.astype('uint8')
                    es_pred_gt_pad *= roi_mask.astype('uint8')
                    # Saver and nii converter
                    ED = (ed_img, ed_gt, ed_pred_gt_pad)
                    ES = (es_img, es_gt, es_pred_gt_pad)
                    data = (ED, ES)
                elif approach==2:
                    # seems that the results of predicting batchwise and slicewise differs at apex
                    # mostly becuase of BN?? but its good to do prediction on batchwise
                    ed_patch_img, pad_params = extract_patch(ed_img, roi_center, patch_size=128)
                    es_patch_img, _ = extract_patch(es_img, roi_center, patch_size=128)           

                    ed_pred_gt = self.predict_feeder(ed_patch_img)
                    print('Patient-ED', pid)
                    es_pred_gt = self.predict_feeder(es_patch_img)
                    print('Patient-ES', pid)
                    # pad the predictions and translate
                    ed_pred_gt_pad = pad_patch(ed_pred_gt, pad_params)
                    es_pred_gt_pad = pad_patch(es_pred_gt, pad_params)
                    # Post processing mask 
                    ed_pred_gt_pad *= roi_mask.astype('uint8')
                    es_pred_gt_pad *= roi_mask.astype('uint8')
                    # Saver and nii converter
                    ED = (ed_img, ed_gt, ed_pred_gt_pad)
                    ES = (es_img, es_gt, es_pred_gt_pad)
                    data = (ED, ES)
                else:
                    # Predict the ouput as it is 
                    ed_pred_gt = self.predict_feeder(ed_img)
                    print('Patient-ED', pid)

                    es_pred_gt = self.predict_feeder(es_img)
                    print('Patient-ES', pid)

                    # Post processing mask
                    ed_pred_gt *= roi_mask.astype('uint8')
                    es_pred_gt *= roi_mask.astype('uint8')
                    # Saver and nii converter
                    ED = (ed_img, ed_gt, ed_pred_gt)
                    ES = (es_img, es_gt, es_pred_gt)
                    data = (ED, ES)
                save_prediction(data, pid, affine, hdr, roi_mask, path=path)
            except Exception,e:
                print str(e)
        return