import os
import tensorflow as tf
import shutil
# from datetime import datetime as dt


class SummaryManager(object):
	def __make_summary_dirs(self,logdir):
		self.summary_paths = {
		'train' : os.path.join(logdir,'train'),
		'val' : os.path.join(logdir,'val'),
		'test' : os.path.join(logdir,'test')
		}
		for key,val in self.summary_paths.items():
			if not os.path.isdir(val): os.makedirs(val)

	def __init__(self,logdir,graph,frequencies):
		self.__make_summary_dirs(logdir)
		self.train_writer = tf.summary.FileWriter(
			self.summary_paths['train'],graph)
		self.val_writer = tf.summary.FileWriter(
			self.summary_paths['val'],graph)
		self.test_writer = tf.summary.FileWriter(
			self.summary_paths['test'],graph)

		self.train_counts = {}
		self.val_counts = {}
		self.test_counts = {}

		for key in frequencies:
			self.train_counts[key] = 0
			self.val_counts[key] = 0
			self.test_counts[key] = 0

	def add_summary(self,summary,mode,freq_key):
		if mode == 'train':
			count = self.train_counts[freq_key]
			self.train_writer.add_summary(summary,count)
			self.train_counts[freq_key] = self.train_counts[freq_key] + 1
		elif mode == 'val':
			count = self.val_counts[freq_key]
			self.val_writer.add_summary(summary,count)
			self.val_counts[freq_key] = self.val_counts[freq_key] + 1
		elif mode == 'test':
			count = self.test_counts[freq_key]
			self.test_writer.add_summary(summary,count)
			self.test_counts[freq_key] = self.test_counts[freq_key] + 1
		else:
			print('Error : mode is not one of \'train\',\'val\' or \'test\'')

	def update_counts(self,count_dict):
		for key in count_dict['train']:
			self.train_counts[key] = count_dict['train'][key] 
		for key in count_dict['val']:
			self.val_counts[key] = count_dict['val'][key]
		for key in count_dict['test']:
			self.test_counts[key] = count_dict['test'][key]