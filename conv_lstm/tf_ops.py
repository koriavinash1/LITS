from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
from conv_lstm_gru.cell import ConvLSTMCell
from conv_lstm_gru.cell import ConvGRUCell

def conv_rnn(inputs, filters=None, kernel_size=[3, 3], cell_type='GRU', name=None):
    """
    Implementation of conv-rnn for replacing skip-connection link
    Put this as module between up-sampling and down-sampling layers, 
    ideally this function is for replacing skip connection with recurrent connection
    which preserves the history of previous slices

    This module currently accepts and outputs data of format:
    Args: 

    Input:  
        Batch_Size * Height * Width * Channels
    Outputs:
        Batch_Size * Height * Width * Channels
    Here : Batch_size: -> Consecutive slices
    TODO: 
    Currently the netork is designed to accept 1-patient volume and split the volume into chunks of batch size
    The LSTM/GRU cell accepts inputs in Time-Major format, i.e. Time_Steps(here slices)*Batch_Size(here 1-patient)* H*W*channels         
    """

    # inputs are oriinally in Slices*H*W*C format so convert to
    # Slices*1*H*W*C (converting to time =-major format)
    patient_batch_size = 1 
    inputs = tf.expand_dims(inputs, axis=1, name=name+'/expand_dims')
    # print (inputs.shape)
    # Shape of Height and width of the feature map
    shape = [inputs.get_shape().as_list()[2], inputs.get_shape().as_list()[3]]
    if filters is None:
        # Number of filters to learn depends on the size of input feature map channels
        filters = inputs.get_shape().as_list()[4]

    if cell_type == 'GRU':
        cell = ConvGRUCell(shape, filters, kernel_size)
        state_var = tf.get_variable(name+'/state_var', shape=[patient_batch_size]+shape+[filters], 
                    trainable=False, initializer=tf.zeros_initializer())
        # print (state_var.shape)
    elif cell_type == 'LSTM':
        c_state_var = tf.get_variable(name+'/c_state_var', shape=[patient_batch_size]+shape+[filters], 
                        trainable=False, initializer=tf.zeros_initializer())
        h_state_var = tf.get_variable(name+'/h_state_var', shape=[patient_batch_size]+shape+[filters], 
                        trainable=False, initializer=tf.zeros_initializer())
        state_var = tf.nn.rnn_cell.LSTMStateTuple(c_state_var, h_state_var)
        cell = ConvLSTMCell(shape, filters, kernel_size, peephole=False) 

    outputs, states = tf.nn.dynamic_rnn(cell, inputs, sequence_length=None,
                                        initial_state=state_var,
                                        dtype=inputs.dtype,
                                        parallel_iterations=None,
                                        swap_memory=False,
                                        time_major=True,
                                        scope=name+'/conv_rnn')
    # Squueze out the iniitally expanded dimension (i.e. batch_size axis)
    outputs = tf.squeeze(outputs, name=name+'/squeeze_dims')
    if cell_type == 'GRU':
        state_update_ops = tf.assign(state_var, states)
        reset_ops = tf.variables_initializer([state_var], name=name+'/reset_state')
    elif cell_type == 'LSTM':     
        c_state_var_ops = tf.assign(c_state_var, states[0])
        h_state_var_ops = tf.assign(h_state_var, states[1])
        state_update_ops = tf.group(c_state_var_ops, h_state_var_ops)
        reset_ops = tf.variables_initializer([c_state_var, h_state_var], name=name+'/reset_state')
    print("Convolutional RNN:", name, "shape ", outputs.get_shape().as_list())
    return outputs, state_update_ops, reset_ops  



def conv2d(inputs, filters, kernel_size, name=None, activation_fn=None):
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
            mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
    # initializer=None
    outputs = tf.layers.conv2d(inputs,
                filters,
                kernel_size,
                strides=(1, 1),
                padding='same',
                data_format='channels_last',
                dilation_rate=(1, 1),
                activation=activation_fn,
                use_bias=True,
                kernel_initializer=initializer,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=name,
                reuse=None
            )
    return outputs

def batch_norm(inputs, is_training, name=None):
    """
    Note: when training, the moving_mean and moving_variance need to be updated.
    By default the update ops are placed in tf.GraphKeys.UPDATE_OPS,
    so they need to be added as a dependency to the train_op. For example:

      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

    """
    output = tf.layers.batch_normalization(
                        inputs,
                        axis=-1,
                        momentum=0.99,
                        epsilon=0.001,
                        center=True,
                        scale=True,
                        beta_initializer=tf.zeros_initializer(),
                        gamma_initializer=tf.ones_initializer(),
                        moving_mean_initializer=tf.zeros_initializer(),
                        moving_variance_initializer=tf.ones_initializer(),
                        beta_regularizer=None,
                        gamma_regularizer=None,
                        training=is_training,
                        trainable=True,
                        name=name,
                        reuse=None,
                        renorm=False,
                        renorm_clipping=None,
                        renorm_momentum=0.99
                    )

    return output

def BN_ReLU_Conv(inputs, n_filters, is_training, name, filter_size=(3,3), dropout_p=0.1):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, 
    Convolution and Dropout (if dropout_p > 0) on the inputs
    """
    outputs = inputs
    outputs = batch_norm(outputs, is_training, name=name+'/bn')
    outputs = tf.nn.relu(outputs, name=name+'/relu')
    outputs = conv2d(outputs, n_filters, filter_size, name=name+'/conv', activation_fn=tf.nn.relu)
    # batch_size = tf.shape(inputs)[0]
    # noise_shape = tf.concat([batch_size])  
    outputs = tf.layers.dropout(outputs,
                rate=dropout_p,
                noise_shape=None,
                seed=None,
                training=is_training,
                name=name+'/dropout')
    return outputs

def TransitionDown(inputs, n_filters, is_training, name, dropout_p=0.1):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, 
        and a max pooling with a factor 2  """

    outputs = BN_ReLU_Conv(inputs, n_filters, is_training, name+'/bn-relu-conv', 
                        filter_size=(1,1), 
                        dropout_p=dropout_p)
    outputs = tf.layers.max_pooling2d(outputs,
                                    pool_size=(2,2),
                                    strides=(2,2),
                                    padding='valid',
                                    data_format='channels_last',
                                    name=name+'/max_pool2d')
    return outputs

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep, name, channel_axis,
                filter_size=(3,3)):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates 
    it with the skip_connection """

    outputs = tf.concat(block_to_upsample, axis=channel_axis, name=name+'/concat')
    # Upsample
    # TODO: Check padding scheme
    padding = 'same' if 2*int(outputs.shape[1])==skip_connection.shape[1] else 'valid'
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
                mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
    outputs = tf.layers.conv2d_transpose(outputs,
                n_filters_keep,
                kernel_size=filter_size,
                strides=(2, 2),
                padding=padding,
                data_format='channels_last',
                activation=tf.nn.relu,
                use_bias=True,
                kernel_initializer=initializer,
                bias_initializer=tf.zeros_initializer(),
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                trainable=True,
                name=name+'/transpose_conv2d_relu',
                reuse=None)
    # Concatenate with skip connection
    outputs = tf.concat([outputs, skip_connection], axis=channel_axis, name=name+'/concat')
    return outputs

def FinalLayer(inputs, n_classes, name):
    """
    Performs 1x1 convolution followed by relu nonlinearity
    """
    outputs = conv2d(inputs, n_classes, kernel_size=(1,1), name=name+'/conv_1_1',
              activation_fn=tf.nn.relu)
    return outputs

def _conv2d(inputs, num_outputs, kernel_size, scope, norm=True,
           d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d(
        inputs, num_outputs, kernel_size, scope=scope,
        data_format=d_format, activation_fn=None, biases_initializer=None)
    if norm:
        outputs = tf.contrib.layers.batch_norm(
            outputs, decay=0.9, center=True, activation_fn=tf.nn.relu,
            updates_collections=None, epsilon=1e-5, scope=scope+'/batch_norm',
            data_format=d_format)
    else:
        outputs = tf.nn.relu(outputs, name=scope+'/relu')
    return outputs


def co_conv2d(inputs, out_num, kernel_size, scope, norm=True,
              d_format='NHWC'):
    conv1 = tf.contrib.layers.conv2d(
        inputs, out_num, kernel_size, stride=2, scope=scope+'/conv0',
        data_format=d_format, activation_fn=None, biases_initializer=None)
    outputs = dilated_conv(conv1, out_num, kernel_size, scope)
    return outputs


def deconv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    outputs = tf.contrib.layers.conv2d_transpose(
        inputs, out_num, kernel_size, scope=scope, stride=[2, 2],
        data_format=d_format, activation_fn=None, biases_initializer=None)
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def co_dilated_conv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    axis = (d_format.index('H'), d_format.index('W'))
    channel_axis = d_format.index('C')
    conv1 = conv2d(inputs, out_num, kernel_size, scope+'/conv1', False)
    conv1_concat = tf.concat(
        [inputs, conv1], channel_axis, name=scope+'/concat1')
    conv2 = conv2d(conv1_concat, out_num, kernel_size, scope+'/conv2', False)
    conv2_concat = tf.concat(
        [conv1_concat, conv2], channel_axis, name=scope+'/concat2')
    conv3 = conv2d(conv2_concat, 2*out_num, kernel_size, scope+'/conv3', False)
    conv4, conv5 = tf.split(conv3, 2, channel_axis, name=scope+'/split')
    dialte1 = dilate_tensor(conv1, axis, 0, 0, scope+'/dialte1')
    dialte2 = dilate_tensor(conv2, axis, 1, 1, scope+'/dialte2')
    dialte3 = dilate_tensor(conv4, axis, 1, 0, scope+'/dialte3')
    dialte4 = dilate_tensor(conv5, axis, 0, 1, scope+'/dialte4')
    outputs = tf.add_n([dialte1, dialte2, dialte3, dialte4], scope+'/add')
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def dilated_conv(inputs, out_num, kernel_size, scope, d_format='NHWC'):
    axis = (d_format.index('H'), d_format.index('W'))
    conv1 = conv2d(inputs, out_num, kernel_size, scope+'/conv1', False)
    dilated_inputs = dilate_tensor(inputs, axis, 0, 0, scope+'/dialte_inputs')
    dilated_conv1 = dilate_tensor(conv1, axis, 1, 1, scope+'/dialte_conv1')
    conv1 = tf.add(dilated_inputs, dilated_conv1, scope+'/add1')
    with tf.variable_scope(scope+'/conv2'):
        shape = list(kernel_size) + [out_num, out_num]
        weights = tf.get_variable(
            'weights', shape, initializer=tf.truncated_normal_initializer())
        weights = tf.multiply(weights, get_mask(shape, scope))
        strides = [1, 1, 1, 1]
        conv2 = tf.nn.conv2d(conv1, weights, strides, padding='SAME',
                             data_format=d_format)
    outputs = tf.add(conv1, conv2, name=scope+'/add2')
    return tf.contrib.layers.batch_norm(
        outputs, decay=0.9, activation_fn=tf.nn.relu, updates_collections=None,
        epsilon=1e-5, scope=scope+'/batch_norm', data_format=d_format)


def get_mask(shape, scope):
    new_shape = (shape[0]*shape[1], shape[2], shape[3])
    mask = np.ones(new_shape, dtype=np.float32)
    for i in range(0, new_shape[0], 2):
        mask[i, :, :] = 0
    mask = np.reshape(mask, shape, 'F')
    return tf.constant(mask, dtype=tf.float32, name=scope+'/mask')


def dilate_tensor(inputs, axis, row_shift, column_shift, scope):
    rows = tf.unstack(inputs, axis=axis[0], name=scope+'/rowsunstack')
    row_zeros = tf.zeros(
        rows[0].shape, dtype=tf.float32, name=scope+'/rowzeros')
    for index in range(len(rows), 0, -1):
        rows.insert(index-row_shift, row_zeros)
    row_outputs = tf.stack(rows, axis=axis[0], name=scope+'/rowsstack')
    columns = tf.unstack(
        row_outputs, axis=axis[1], name=scope+'/columnsunstack')
    columns_zeros = tf.zeros(
        columns[0].shape, dtype=tf.float32, name=scope+'/columnzeros')
    for index in range(len(columns), 0, -1):
        columns.insert(index-column_shift, columns_zeros)
    column_outputs = tf.stack(
        columns, axis=axis[1], name=scope+'/columnsstack')
    return column_outputs

def pool2d(inputs, kernel_size, scope, data_format='NHWC'):
    return tf.contrib.layers.max_pool2d(
        inputs, kernel_size, scope=scope, padding='SAME',
        data_format=data_format)

def weight_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)