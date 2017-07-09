import tensorflow as tf
import collections

class NpTfMap(object):
	""" for easy numpy/python data to tensorflow variable handling"""
	def __init__(self):
		self.var_dict = {}

	def init_elements(self,val_dict):
		for key,val in val_dict.items():
			self.__dict__[key] = val	
		def create_variables(var_dict,val_dict):
			for key,val in val_dict.items():
				if isinstance(val,collections.Mapping):
					r = create_variables(var_dict.get(key,{}),val)
					var_dict[key] = r
				else:
					var_dict[key] = tf.Variable(val_dict[key],name = key,trainable = False)
			return var_dict
		create_variables(self.var_dict,val_dict)

	def update_elements(self,var_dict):
		for key, val in var_dict.items():
			self.__dict__[key] = val

	def assign_ops(self):
		assign_ops = []
		def get_assign_ops(var_dict,val_dict):
			for key,val in var_dict.items():
				if isinstance(val,collections.Mapping):
					get_assign_ops(var_dict[key],val_dict[key])
				else:
					assign_ops.append(tf.assign(var_dict[key],val_dict[key]))
		get_assign_ops(self.var_dict,self.__dict__)
		return assign_ops
