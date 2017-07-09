import argparse
import pprint

default_opts = argparse.ArgumentParser(add_help=False)

default_opts.add_argument('--num_epochs', 
	default = 200, type = int, help = 'Number of Epochs ')
default_opts.add_argument('--batch_size',
	default = 2, type = int, help = 'Batch Size')
default_opts.add_argument('--num_class',
	default = 3, type = int, help = 'Number of classes')
default_opts.add_argument('--num_channels',
	default = 3, type = int, help = 'Number of channels')
default_opts.add_argument('--chief_class',
	default = 1, type = int, help = 'Chief class for dice_score')
default_opts.add_argument('--num_gpus',
	default = 1, type = int, help = 'Number of gpus')

default_opts.add_argument('--data_folder',default = '/windows/Lits2017/Dataset/hdf5/', help = 'data folder')

default_opts.add_argument('--output_dir', default = '/windows/Lits2017/outputs/', help = 'output directory for models and summaries')
default_opts.add_argument('--run_name', default= 'tiramisu_liver_tumor_edge_weights_3channel_512_out', help = 'model and summaries will be saved in a folder with this name')
default_opts.add_argument('--resume_training', default = False, type = bool, help = 'resume training or not ?')
default_opts.add_argument('--load_model_from', default= None, help='specify directory from which to load model from')
default_opts.add_argument('--pretrain_model',default = False, type = bool, help = 'True if restoring from a pretrained model')

# '/home/kiran/axon/saved_models/tiramisu_tumor_axial/best_model/latest.ckpt'
# this opt builder is for ipython
class _OPT:
	def __init__(self):
		default, unknown = default_opts.parse_known_args()
		self.__dict__.update(vars(default))

	def update(self,**kwargs):
		self.__dict__.update(kwargs)

	def __repr__(self):
		return pprint.pformat(self.__dict__,width = 1)

	def __str__(self):
		return pprint.pformat(self.__dict__,width = 1)

opts = _OPT()


# debug mode or not 
# if not in debug mode, then prompt with questions 
# start a new run directory ?
# archive previous run ?

# resume_training should not ask for new run directory 
# 
# run_name
# 
