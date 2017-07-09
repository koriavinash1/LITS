from estimator import *
import argparse
import os, sys, shutil
import numpy as np 
import time
sys.path.insert(0,'../utils/')
from train_utils import *

# sys.path.insert(0,'../net/')
# from net_2d import Tiramisu

if __name__ == '__main__':
	parser = argparse.ArgumentParser(parents = [default_opts])
	opts = parser.parse_args()
	run_dir = os.path.join(opts.output_dir,opts.run_name)
	model_path = os.path.join(run_dir,'models','latest.ckpt')

	if opts.resume_training and (opts.load_model_from == None):
		opts.load_model_from = model_path

	summary_dir = os.path.join(opts.output_dir,opts.run_name,'summary')
	summary_manager = SummaryManager(summary_dir,tf.get_default_graph(),
		frequencies = ['per_step','per_100_steps','per_epoch','per_five_epochs'])

	inputs = Inputs(None,None,None,opts.num_channels)
	targets = Targets(None,None,None,opts.num_class)
	weight_maps = tf.placeholder(tf.float32, shape=[None,None,None])

	# define the net
	print('Defining the network')
	net = Tiramisu(inputs,
					targets, 
					weight_maps,
					n_pool=5,
					n_feat_first_layer=48,
					growth_rate=16, 
					n_layers_per_block=5,
					chief_class = opts.chief_class,
					weight_decay = 5e-6, 
					keep_prob = 0.8, 
					metrics_list = ['loss', 'dice_score'],
					optimizer = Adam(1e-4),
					gpu_ids = [0])
	
	# initialise the estimator with the net 	
	print('Preparing the estimator..')
	trainer = Estimator(net_obj = net, 
						summary_manager = summary_manager,
						resume_training = opts.resume_training, 
						load_model_from = opts.load_model_from, 
						save_path = model_path)

	# iterate for the number of epochs
	for epoch in range(trainer.np_tf_map.epoch, opts.num_epochs):
		print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
		print('Training @ Epoch : ' + str(epoch))
		trainer.fit()
		print('\n---------------------------------------------------')
		print('Validating @ Epoch : ' + str(epoch))
		trainer.evaluate()
		trainer.saveModel(os.path.join(opts.output_dir,opts.run_name,'models','latest.ckpt'))
		trainer.np_tf_map.epoch = trainer.np_tf_map.epoch + 1
