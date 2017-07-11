from __future__ import division
import numpy as np
import os, sys, shutil
import tensorflow as tf
sys.path.insert(0,'../utils/')
from train_utils import *

class Tiramisu(object):
	def __init__(self, 
				inputs, 
				targets, 
				weight_maps, 
				n_pool = 3, 
				n_feat_first_layer = 48, 
				growth_rate = 16,
				n_layers_per_block = 3, 
				chief_class = 1,
				weight_decay = 5e-6, 
				keep_prob = 0.8, 
				metrics_list = ['loss', 'dice_score'],
				metric_to_optimize = 'loss',
				optimizer = Adam(1e-4),
				gpu_ids =[1,2]):
		self.inputs = inputs
		self.targets = targets
		self.weight_maps = weight_maps
		self.n_pool = n_pool
		self.n_feat_first_layer = n_feat_first_layer
		self.growth_rate = growth_rate
		self.n_layers_per_block = n_layers_per_block
		self.chief_class = chief_class
		self.weight_decay = weight_decay
		self.keep_prob = keep_prob
		self.metrics_list = metrics_list
		self.optimizer = optimizer
		self.gpu_ids = gpu_ids
		self.num_gpus = len(self.gpu_ids)
		self.metric_to_optimize = metric_to_optimize

		self.image_splits = tf.split(self.inputs,self.num_gpus,0)
		self.labels_splits = tf.split(self.targets,self.num_gpus,0)
		self.weight_splits = tf.split(self.weight_maps,self.num_gpus,0)

		self.is_training = tf.placeholder(tf.bool)
		self.num_channels = inputs.get_shape()[-1].value
		self.num_classes = targets.get_shape()[-1].value

		self.logits = {}
		self.posteriors = {}
		self.predictions = {}


		with tf.variable_scope(tf.get_variable_scope()) as vscope:
			for i in gpu_ids:
				idx = gpu_ids.index(i)
				with tf.name_scope('tower_%d'%idx):
					with tf.device('/gpu:%d'%i):
						self.doForward(idx,
										n_feat_first_layer = self.n_feat_first_layer,
										n_pool = self.n_pool,
										growth_rate = self.growth_rate,
										n_layers_per_block = self.n_layers_per_block,
										keep_prob = self.keep_prob
										)
						tf.get_variable_scope().reuse_variables()


	def doForward(self,idx, n_feat_first_layer, n_pool, growth_rate, n_layers_per_block, keep_prob):
		
		assert type(n_layers_per_block) == int
		n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)

		inputs = self.image_splits[idx]

		with tf.variable_scope('input_layer'):
			stack = Conv2D(inputs, [3,3,inputs.get_shape()[-1].value,n_feat_first_layer],collection_name = 'input_layer', padding='SAME')

		n_filters = n_feat_first_layer

		skip_connection_list = []

		#######################
		#   Downsampling path   #
		#######################

		for i in range(n_pool):
			for j in range(n_layers_per_block[i]):
				with tf.variable_scope('dense' + str(i) + '_' + str(j)):
					l = BN_eLU_Conv(stack, growth_rate,collection_name = 'block_%d_layer_%d'%(i,j), keep_prob=keep_prob, is_training=self.is_training)
					stack = tf.concat([stack, l],3)
				n_filters += growth_rate

			skip_connection_list.append(stack)

			with tf.variable_scope('dense' + str(i) + '_TD'):
				stack = TransitionDown(stack, n_filters,collection_name = 'transition_%d'%(i),keep_prob=keep_prob, is_training=self.is_training)

		skip_connection_list = skip_connection_list[::-1]
		block_to_upsample = []

		for j in range(n_layers_per_block[n_pool]):
			with tf.variable_scope('densemid' + str(j)):
				l = BN_eLU_Conv(stack, growth_rate, collection_name = 'bottle_neck_layer', keep_prob=keep_prob, is_training=self.is_training)
				block_to_upsample.append(l)
				stack = tf.concat([stack, l],3)

		#######################
		#   Upsampling path   #
		#######################

		for i in range(n_pool):
			with tf.variable_scope('transitionup' + str(i)):
				n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
				stack = TransitionUp(skip_connection_list[i], block_to_upsample, n_filters_keep,collection_name = 'transition_%d'%(n_pool-i-1), is_training=self.is_training)

			block_to_upsample = []
			for j in range(n_layers_per_block[n_pool + i + 1]):
				with tf.variable_scope('transitionup' + str(i) + '_dense' + str(j)):
					l = BN_eLU_Conv(stack, growth_rate, collection_name = 'block_%d_layer_%d'%(n_pool-i-1,j), keep_prob=keep_prob, is_training=self.is_training)
					block_to_upsample.append(l)
					stack = tf.concat([stack, l],3)

		#######################
		#   More upsampling   #
		#######################
		l = Conv2D(stack,[3,3,stack.get_shape()[-1].value,4],collection_name = 'up_conv_512', padding = 'SAME')
		stack = SpatialBilinearUpsampling(l,factor = 2)
		l = None

		#####################
		# 		Outputs 	#
		#####################

		with tf.variable_scope('logits'):
			self.logits[idx] = Conv2D(stack, [3,3,stack.get_shape()[-1].value,self.num_classes], collection_name = 'logits_layer', padding='SAME')

		with tf.variable_scope('logits'):
			self.posteriors[idx] = Softmax(self.logits[idx])	

		with tf.variable_scope('predictions'):
			self.predictions[idx] = tf.cast(tf.argmax(self.logits[idx],3), tf.float32)
