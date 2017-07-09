from __future__ import division
import tensorflow as tf 
import numpy as np 
import os, sys, shutil
import pprint
import time, argparse
sys.path.insert(0,'../net/')
from 2d_net import Tiramisu
sys.path.insert(0,'../data_loader/')
from data_loader import ITERATOR
sys.path.insert(0,'../utils/')
from train_utils import *
from nptfmap import NpTfMap
from summary_manager import SummaryManager
from opts import opts

class Estimator(object):
	def __init__(self,
				 net_obj,
				 summary_manager,
				 resume_training,
				 load_model_from,
				 save_path):
		
		self.net = net_obj
		self.data_iterator = ITERATOR(opts.data_folder, opts.batch_size, n_class=2, num_threads=1)
		print('preparing summary manager')
		self.summary_manager = summary_manager
		print('preparing nptfmaps')
		self.np_tf_map = NpTfMap()
		
		#initalising np_tf_map
		self.np_tf_map.init_elements({'epoch':0, 'dice_score':0.0})
		self.np_tf_map.init_elements({'train_counts' : self.summary_manager.train_counts, 'val_counts' : self.summary_manager.val_counts,
			'test_counts' : self.summary_manager.test_counts})
		
		print('defining the session')
		self.sess = utils.flexiSession()
		self.sess.run(tf.global_variables_initializer())
		
		try:
			self.sess.run(tf.assert_variables_initialized())
		except tf.errors.FailedPreconditionError:
			raise RuntimeError('Not all variables initialized')

		# saver to resetore
		self.saver = tf.train.Saver(tf.global_variables())

		# check whether to restore or not from the args
		if opts.pretrain_model and (opts.load_model_from is not None):
			self.net.restoreFromPreTrainedModel(self.sess,load_model_from)

		elif opts.pretrain_model and (opts.load_model_from is None):
			raise RuntimeError("pretrain_model path not given at opts.load_model_from !!!")

			
		if opts.load_model_from is None:
			print("load_model_from is None")
	
		if ((resume_training == True) or (load_model_from is not None)) and (not opts.pretrain_model):
			self.restoreModel(load_model_from)

	def restoreModel(self,load_model_from):
		print('Restoring model from: ' + str(load_model_from))
		self.saver.restore(self.sess,load_model_from)
		self.np_tf_map.update_elements(self.sess.run(self.np_tf_map.var_dict))
		count_dict = {'train' : self.np_tf_map.train_counts,
					  'val'   : self.np_tf_map.val_counts,
					  'test'  : self.np_tf_map.test_counts }
		self.summary_manager.update_counts(count_dict)
		print('Epochs completed : ' + str(self.np_tf_map.epoch))
		print('Best dice: ' + str(self.np_tf_map.dice_score))

	def saveModel(self,save_path):
		self.sess.run(self.np_tf_map.assign_ops())
		model_dir = os.path.split(save_path)[0]
		if not os.path.isdir(model_dir): 
			os.makedirs(model_dir)
		self.saver.save(self.sess, save_path)


	def fit(self, steps=1000):
		self.data_iterator.mode = 'train'
		self.data_iterator.reset()
		time.sleep(5.0)

		feed = None
		train_ops = [self.net.inference_ops,
					 self.net.summary_ops['1step'],
					 self.net.update_op,
					 self.net.accumulate_ops,
					 self.net.logits]
		
		count = 0
		step_sec = 0
		while (count < steps):
			start_time = time.time()

			# fetch inputs batches and verify if they are numpy.ndarray and run all the ops
			input_batch, target_batch, weight_batch = self.data_iterator.getNextBatch()
			
			if type(input_batch) is np.ndarray:

				feed = {self.net.inputs : input_batch,
						self.net.targets : target_batch,
						self.net.weight_maps : weight_batch,
						self.net.is_training : True}
				input_batch, target_batch, weight_batch = None, None, None
				
				inferences, summary, _, __, outputs = self.sess.run(train_ops,feed_dict =  feed)

				self.summary_manager.add_summary(summary,"train","per_step")
				utils.progress(count%250+1,250, step_sec)
				
				inferences, outputs, summary = None, None, None

				# add summaries regularly for every 100 steps
				if (count + 1)% 100 == 0:
					summary = self.sess.run(self.net.summary_ops['100steps'], feed_dict = feed)	
					self.summary_manager.add_summary(summary,"train","per_100_steps")

				if (count + 1) % 250 == 0:
					print('Avg metrics : ')
					pprint.pprint(self.sess.run(self.net.stats_ops), width = 1)
				count = count + 1 

			stop_time = time.time()
			step_sec = stop_time - start_time
			if self.data_iterator.iter_over == True:
				self.data_iterator.reset()
				time.sleep(4)


		print('\nAvg metrics for epoch : ')
		pprint.pprint(self.sess.run(self.net.stats_ops), width = 1)
		summary = self.sess.run(self.net.summary_ops['1epoch'],
			feed_dict = feed)
		self.sess.run(self.net.reset_ops)
		self.summary_manager.add_summary(summary,"train","per_epoch")
		summary = None


	def evaluate(self, steps=250):
		#set mode and wait for the threads to populate the queue
		self.data_iterator.mode = 'val'
		self.data_iterator.reset()
		time.sleep(5.0)
		
		feed = None
		val_ops = [self.net.inference_ops,
					 self.net.summary_ops['1step'],
					 self.net.accumulate_ops,
					 self.net.logits]

		#iterate the validation step, until count = steps
		count = 0
		step_sec = 0
		while count < steps:
			start_time = time.time()
			time.sleep(0.4)

			input_batch, target_batch, weight_batch = self.data_iterator.getNextBatch()
			if type(input_batch) is np.ndarray:

				feed = {self.net.inputs : input_batch,
						self.net.targets : target_batch,
						self.net.weight_maps :  weight_batch,
						self.net.is_training : False}

				input_batch, target_batch, weight_batch = None, None, None
				
				inferences, summary, _, outputs = self.sess.run(val_ops,feed_dict = feed) 
				self.summary_manager.add_summary(summary,"val","per_step")
				utils.progress(count%250+1,250, step_sec)
				inferences, outputs, summary = None, None, None
				
				if (count+1) % 100 == 0:
					summary = self.sess.run(self.net.summary_ops['100steps'], feed_dict = feed)
					self.summary_manager.add_summary(summary,"val",
						"per_100_steps")

				if (count+1) % 250 == 0:
					print('Avg metrics : ')
					pprint.pprint(self.sess.run(self.net.stats_ops), width = 1)
				count = count + 1 
			stop_time = time.time()
			step_sec = stop_time - start_time
			if self.data_iterator.iter_over == True:
				print('\nIteration over')
				self.data_iterator.reset()
				count = steps
				time.sleep(5)

		print('\nAvg metrics for epoch : ')
		metrics = self.sess.run(self.net.stats_ops)
		pprint.pprint(metrics, width=1)
		if (metrics[0]['dice_score'] > self.np_tf_map.dice_score):
			self.np_tf_map.dice_score = metrics[0]['dice_score']

			print('Saving best model!')
			self.saveModel(os.path.join(opts.output_dir,opts.run_name,'best_model','latest.ckpt'))
		print('Current best dice: ' + str(self.np_tf_map.dice_score))
		summary = self.sess.run(self.net.summary_ops['1epoch'],
			feed_dict = feed)
		self.sess.run(self.net.reset_ops)
		self.summary_manager.add_summary(summary,"val","per_epoch")
		summary = None

