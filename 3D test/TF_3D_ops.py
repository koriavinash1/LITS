import tensorflow as tf

def Conv3D(data,filters,kernel_size,name,strides=(1,1,1),activation=tf.nn.elu):
	initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
		mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
	output = tf.layers.conv3d(data,
			filters,
			kernel_size,
			strides=strides,
			padding='same',
			data_format='channels_last',
			dilation_rate=(1, 1, 1),
			activation=activation,
			kernel_initializer=initializer,
			name=name)
	return output

def Deconv3D(data, filters, kernel_size, name, strides=(1,1,1), activation=tf.nn.elu):
	initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, 
		mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
	output = tf.layers.conv3d_transpose(data,
			filters,
			kernel_size,
			strides=strides,
			padding='same',
			data_format='channels_last',
			activation=activation,
			kernel_initializer=initializer,
			name = name)
	return output

def MaxPooling3D(data, name, pool_size=(2,2,2), strides=(1,1,1)):
	output = max_pooling3d(inputs,
			pool_size,
			strides,
			padding='valid',
			data_format='channels_last',
			name=name)
	return output

def BatchNormalization(data,name,training=False,activation=tf.nn.elu):
	output = tf.layeres.batch_normalization(data,
			axis=-1,
			momentum=0.99,
			epsilon=0.001,
			training=training,
			activation_fn =activation,
			name=name)
	return output

def Concatenate(data, name):
	output = tf.concat(data,
			axis=0,
			name=name)
	return output
