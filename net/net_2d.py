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
		self.grads_dict = {}

		self.stats_ops = {}
		self.inference_ops = {}
		self.accumulate_ops = {}
		self.reset_ops = {}

		for g in gpu_ids:
			g = gpu_ids.index(g)
			self.stats_ops[g] = {}
			self.inference_ops[g] = {}
			self.accumulate_ops[g] = []
			self.reset_ops[g] = []

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
						
						self.calMetrics(idx)
						tf.get_variable_scope().reuse_variables()

		self.optimize()
		self.makeSummaries()


	def doForward(self,idx, n_feat_first_layer, n_pool, growth_rate, n_layers_per_block, keep_prob):
		
		assert type(n_layers_per_block) == int
		n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)

		inputs = self.image_splits[idx]
		tf.summary.image('inputs',inputs[:,:,:,1:2], max_outputs = 4, collections = ['per_100_steps'])

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
			tf.summary.image('predictions',self.predictions[idx][:,:,:,None], max_outputs = 4, collections = ['per_100_steps'])

	def calMetrics(self,gpu_id_idx):
		for metric_name in self.metrics_list:
			metric_implemented = False
			if metric_name == 'loss':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = loss = SpatialWeightedCrossEntropyLogits(self.logits[gpu_id_idx], self.labels_splits[gpu_id_idx],self.weight_splits[gpu_id_idx]) 
					#class weights should be in float
					metric_obj = ScalarMetricStream(loss)

					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])					

			elif metric_name == 'dice_score':
				metric_implemented = True
				with tf.variable_scope(metric_name):
					self.inference_ops[gpu_id_idx][metric_name] = dice_score = DiceCriteria2Cls(self.logits[gpu_id_idx],self.labels_splits[gpu_id_idx],chief_class = self.chief_class)
					metric_obj = ScalarMetricStream(dice_score)
					tf.summary.scalar('per-step',metric_obj.op,collections = ['per_step'])
					tf.summary.scalar('epoch-avg',metric_obj.avg,collections = ['per_epoch'])

			else:
				print('Error : ' + metric_name + ' is not implemented')
			
			if metric_implemented == True: 
				try:		
					self.accumulate_ops[gpu_id_idx].append(metric_obj.accumulate)
					self.stats_ops[gpu_id_idx][metric_name] = metric_obj.stats
					self.reset_ops[gpu_id_idx].append(metric_obj.reset)
				except AttributeError:
					pass

		l2_norm  = sum([tf.nn.l2_loss(v) for v in tf.get_collection('l2_norm_vars')])
		total_loss = self.inference_ops[gpu_id_idx][self.metric_to_optimize] + self.weight_decay*l2_norm
		with tf.variable_scope('gradients'):
			grads = self.optimizer.compute_gradients(total_loss)
		
		self.grads_dict[gpu_id_idx] = grads
		grads = None


	def _averageGradients(self,grads_list):
		average_grads = []
		for grad_and_vars in zip(*grads_list):
			grads = []
			for g, _ in grad_and_vars:
				expanded_g = tf.expand_dims(g, 0)
				grads.append(expanded_g)
				expanded_g = None

			grad = tf.concat(grads, 0)
			grad = tf.reduce_mean(grad, 0)

			v = grad_and_vars[0][1]
			grad_and_var = (grad, v)
			average_grads.append(grad_and_var)
			grad_and_var = None
		return average_grads


	def optimize(self):
		with tf.variable_scope('average_gradients'):
			grads = self._averageGradients(self.grads_dict.values())

		with tf.variable_scope('update_op'):
			self.update_op = self.optimizer.apply_gradients(grads)


	def makeSummaries(self):
		self.summary_ops = {}		
		self.summary_ops['1step'] = tf.summary.merge_all(key = 'per_step')
		self.summary_ops['100steps'] = tf.summary.merge_all(key = 'per_100_steps')	
		self.summary_ops['1epoch'] = tf.summary.merge_all(key = 'per_epoch')
