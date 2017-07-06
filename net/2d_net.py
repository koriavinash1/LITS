from __future__ import division
import numpy as np
import os, sys, shutil
import tensorflow as tf
sys.path.append(0,'../utils/')
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
				weight_decay = 5e-6, 
				keep_prob = 0.6, 
				chief_class = 1,
				metrics_list = ['loss', 'dice'],
				optimizer = Adam(1e-4),
				gpu_ids =[1,2]
				):
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

		self.image_splits = tf.split(self.inputs,self.num_gpus,0)
		self.labels_splits = tf.split(self.targets,self.num_gpus,0)
		self.weight_splits = tf.split(self.weight_maps,self.num_gpus,0)

		with tf.variable_scope(tf.get_variable_scope()) as vscope:
			for i in gpu_ids:
				idx = gpu_ids.index(i)
				with tf.name_scope('tower_%d'%idx):
					with tf.device('/gpu:%d'%i):
						self.doForward(n_feat_first_layer = n_feat_first_layer,
										n_pool = n_pool,
										growth_rate = growth_rate,
										n_layers_per_block = n_layers_per_block,
										keep_prob = keep_prob
										)
						
						self.calMetrics(idx)
						tf.get_variable_scope().reuse_variables()

		self.optimize()
		self.addSummaries()


	def doForward(n_feat_first_layer = n_feat_first_layer, n_pool = n_pool, growth_rate = growth_rate, n_layers_per_block = n_layers_per_block, keep_prob = keep_prob):
		