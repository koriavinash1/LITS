import tensorflow as tf
import numpy as np
import sys, os, shutil

def __Weight_Bias(W_shape, b_shape):
    with tf.device('/cpu:0'):
        W = tf.get_variable(name = 'weights', shape = W_shape,initializer = tf.truncated_normal_initializer(stddev=0.1/np.prod(W_shape),dtype = tf.float32))
        tf.add_to_collection('l2_vars',W)
        b = tf.get_variable(name = 'biases', shape = b_shape, initializer = tf.constant_initializer(0.1))
        tf.add_to_collection('l2_vars',b)
    return W, b

def Inputs(*args):
  return tf.placeholder(tf.float32,args,name = 'Inputs')
    
def Targets(*args):
  return tf.placeholder(tf.float32,args,name = 'Targets')

def OneHot(targets,num_class):
  return tf.one_hot(targets,num_class,1,0)


def Dropout(x, drop_mode, keep_prob = 0.7):
  keep_prob_pl = tf.cond(drop_mode, lambda : tf.constant(keep_prob), lambda : tf.constant(1.0))
  return tf.nn.dropout(x,keep_prob_pl)

def flexiSession():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    return tf.Session(config = config)

def Conv2D(x, filter_shape, stride = 1, padding = 'VALID', collections = []):
    strides = None
    if isinstance(stride,int):
        strides = [1,stride,stride,1]
    if isinstance(stride,(list,tuple)):
        strides = [1,stride[0],stride[1],1]

    if isinstance(padding,int):
        tf.pad(x,[[0,0],[padding,padding],[padding,padding],[0,0]])
        padding = 'VALID'  

    W_shape = filter_shape
    b_shape = [filter_shape[3]]
    W, b = __Weight_Bias(W_shape, b_shape)
    for c in collections:
        tf.add_to_collection(c, W)
        tf.add_to_collection(c, b)
    conv_out = tf.nn.conv2d(x, W, strides, padding)
    ret_val = conv_out + b
    
    return ret_val

def Elu(x):
    return tf.nn.elu(x)

def MaxPool2(x):
    ret_val = tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')
    return ret_val

def BatchNorm(inputs, bn_mode, decay = 0.9, epsilon=1e-3, collections = []):

    with tf.device('/cpu:0'):
        scale = tf.get_variable(name = 'scale', shape = inputs.get_shape()[-1], 
            initializer = tf.constant_initializer(1.0),dtype = tf.float32)
        tf.add_to_collection('l2_norm_vars', scale)
        beta = tf.get_variable(name = 'beta', shape = inputs.get_shape()[-1], 
            initializer = tf.constant_initializer(0.0),dtype = tf.float32)
        tf.add_to_collection('l2_norm_vars',beta)
 
    pop_mean = tf.get_variable(name = 'pop_mean', shape = inputs.get_shape()[-1], 
                    initializer = tf.constant_initializer(0.0))
    pop_var = tf.get_variable(name = 'pop_var', shape = inputs.get_shape()[-1],
                    initializer = tf.constant_initializer(1.0))
    tf.summary.histogram('pop_mean', pop_mean, collections = ['per_step'])
    tf.summary.histogram('pop_var', pop_var, collections = ['per_step'])

    for c in collections:
        tf.add_to_collection(c, scale)
        tf.add_to_collection(c, beta)
        tf.add_to_collection(c, pop_mean)
        tf.add_to_collection(c, pop_var)

    axis = list(range(len(inputs.get_shape())-1))

    def Train(inputs, pop_mean, pop_var, scale, beta):
        batch_mean, batch_var = tf.nn.moments(inputs,axis)
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    def Eval(inputs, pop_mean, pop_var, scale, beta):
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    return tf.cond(bn_mode, lambda: Train(inputs, pop_mean, pop_var, scale, beta),
        lambda: Eval(inputs, pop_mean, pop_var, scale, beta))

def ConvEluBatchNormDropout(x, shape, stride = 1,padding = 'VALID', bn_mode = tf.placeholder_with_default(False, shape = []), drop_mode = tf.placeholder_with_default(False, shape = []), keep_prob = 0.7, collections = []):
    return Dropout(BatchNorm(ConvElu(x,shape,stride,padding, collections = collections),bn_mode, collections = collections), drop_mode,keep_prob)

def TransitionDown(inputs, n_filters,collection_name, keep_prob=0.8, is_training=tf.constant(False,dtype=tf.bool)):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_eLU_Conv(inputs, n_filters,collection_name=collection_name, filter_size=1, keep_prob=keep_prob, is_training=is_training)
    l = MaxPool2(l)

    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep,collection_name, is_training=tf.constant(False,dtype=tf.bool)):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """
    l = tf.concat(block_to_upsample,3)
    l = SpatialBilinearUpsampling(l)
    l = BatchNorm(Conv2D(l, [3,3,l.get_shape()[-1].value,n_filters_keep],collection_name = collection_name, padding='SAME'), is_training)
    l = tf.concat([l, skip_connection],3)

    return l

def BN_eLU_Conv(inputs, n_filters,collection_name, filter_size=3, keep_prob=0.8, is_training=tf.constant(False,dtype=tf.bool)):
    l = Elu(BatchNorm(inputs,is_training=is_training))
    l = Conv2D(l, [filter_size, filter_size, l.get_shape()[-1].value, n_filters],collection_name = collection_name, padding='SAME')
    l = Dropout(l, is_training=is_training,keep_prob=keep_prob)
    return l

def SpatialWeightedCrossEntropyLogits(logits, targets, weight_map, name='spatial_wx_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = targets,logits = logits)
    weighted_cross_entropy = tf.multiply(cross_entropy, weight_map)
    mean_weighted_cross_entropy = tf.reduce_mean(weighted_cross_entropy, name=name)
    return mean_weighted_cross_entropy

def DiceCriteria2Cls(logits, targets, chief_class, smooth = 1.0, name = 'dice_score'):
    last_dim_idx = logits.get_shape().ndims - 1
    num_class = tf.shape(logits)[last_dim_idx]
    predictions = tf.one_hot(tf.argmax(logits,last_dim_idx),num_class)
    preds_unrolled = tf.reshape(predictions,[-1,num_class])[:,chief_class]
    targets_unrolled = tf.reshape(targets,[-1,num_class])[:,chief_class]
    intersection = tf.reduce_sum(preds_unrolled*targets_unrolled)
    ret_val = (2.0*intersection)/(tf.reduce_sum(preds_unrolled)
     + tf.reduce_sum(targets_unrolled) + smooth)
    ret_val = tf.identity(ret_val,name = 'dice_score')
    return ret_val

class ScalarMetricStream(object):
    def __init__(self,op, filter_nan = False):
        self.op = op
        count = tf.constant(1.0)
        self.sum = tf.Variable([0.0,0], name = op.name[:-2] + '_sum', trainable = False)
        self.avg = tf.Variable(0.0, name = op.name[:-2] + '_avg', trainable = False)

        if filter_nan == True:
            op_is_nan = tf.is_nan(self.op)
            count = tf.cond(op_is_nan, lambda : tf.constant(0.0), lambda : tf.constant(1.0))
            self.op = tf.cond(op_is_nan, lambda : tf.constant(0.0),lambda : tf.identity(self.op))
        self.accumulate = tf.assign_add(self.sum,[self.op,count])
        self.reset = tf.assign(self.sum,[0.0,0.0])
        self.stats = tf.assign(self.avg,self.sum[0]/(0.001 + self.sum[1]))

def Adam(lr):
    return tf.train.AdamOptimizer(learning_rate = lr)

def progress(curr_idx, max_idx, time_step,repeat_elem = "_"):
    max_equals = 55
    step_ms = int(time_step*1000)
    num_equals = int(iter*max_equals/float(max_idx))
    len_reverse =len('Step:%d ms| %d/%d ['%(step_ms, curr_idx, max_idx)) + num_equals
    sys.stdout.write("Step:%d ms|%d/%d [%s]" %(step_ms, curr_idx, max_idx, " " * max_equals,))
    sys.stdout.flush()
    sys.stdout.write("\b" * (max_equals+1))
    sys.stdout.write(repeat_elem * num_equals)
    sys.stdout.write("\b"*len_reverse)
    sys.stdout.flush()
    if iter == max_idx:
        print('\n')
