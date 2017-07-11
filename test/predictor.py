import numpy as np 
import os, sys, shutil
import scipy.ndimage as snd
import skimage.morphology as morph 
import SimpleITK as sitk
sys.path.insert(0,'../net/')
from net_predictions_2d import Tiramisu
sys.path.insert(0,'../utils/')
from train_utils import *
import train_utils as utils

class Predictor(object):
	def __init__(self, data_folder_path, output_folder_path, model_path, batch_size = 4, num_class = 3, num_channels = 3, gpu_ids = [0]):
		self.data_folder_path = data_folder_path
		self.output_folder_path = output_folder_path
		self.batch_size = batch_size
		self.num_class = num_class
		self.num_channels = num_channels
		self.gpu_ids = gpu_ids

		if not os.path.exists(self.output_folder_path):
			os.makedirs(self.output_folder_path)

		self.erosion_filter = sitk.BinaryErodeImageFilter()
		self.erosion_filter.SetKernelRadius(5)

		self.dilation_filter = sitk.BinaryDilateImageFilter()
		self.dilation_filter.SetKernelRadius(5)

		self.vols, self.segs = self.getFilePaths(self.data_folder_path)
		self.defineNetwork()

		self.sess = utils.flexiSession()

		self.sess.run(tf.global_variables_initializer())

		self.restoreModel(model_path)

	def defineNetwork(self):
		print("Defining the network ... ")
		inputs = Inputs(None,None,None,self.num_channels)
		targets = Targets(None,None,None,self.num_class)
		weights = tf.placeholder(tf.float32,shape=(None,None,None))
		self.net = Tiramisu(inputs,
							targets,
							weights,
							n_pool = 3, 
							n_feat_first_layer = 48, 
							growth_rate = 16,
							n_layers_per_block = 3,
							keep_prob = 1.0,
							gpu_ids = self.gpu_ids
			)
		
		self.saver = tf.train.Saver(tf.global_variables())

	def restoreModel(self,model_path):
		self.saver.restore(self.sess,model_path)

	def getFilePaths(self,data_folder_path,output_folder_path):
		all_fls = os.listdir(data_folder_path)
		vols = []
		segs = []
		for each in all_fls:
			if 'volume-' in each:
				vols.append((os.path.join(data_folder_path,each), os.path.join(output_folder_path,each)))
			elif 'segmentation-' in each:
				segs.append(os.path.join(data_folder_path,each))

		vols.sort()
		segs.sort()
		return vols, segs

	def HU_window(self,in_arr):
	    arr1 = in_arr.copy()
	    arr2 = in_arr.copy()
	    arr3 = in_arr.copy()
	    return np.clip(arr1, 0, 100), np.clip(arr2, -100, 200), np.clip(arr3, -100, 400)

	def histogram_equalization(self,arr):
	    nbr_bins = 256
	    imhist, bins = np.histogram(arr.flatten(), nbr_bins, normed = True)
	    cdf = imhist.cumsum()
	    cdf = 255 * cdf / cdf[-1]
	    original_shape = arr.shape
	    arr = np.interp(arr.flatten(), bins[:-1], cdf)
	    out_arr = arr.reshape(original_shape)
	    return out_arr

	def normalize(self,x):
	    x = np.float32(x)
	    min_val = np.min(x)
	    max_val = np.max(x)
	    ret_val = (x - min_val) / (max_val - min_val)
	    return ret_val

	def downSample(self,data):
		return snd.interpolation.zoom(data,[1.0,0.5,0.5])

	def loadImage(self,path):
		return sitk.ReadImage(vol_path)

	def writeDicomImage(self,data,output_path);
		out_img = sitk.GetImageFromArray(np.uint8(data))
		out_img.SetSpacing(self.img.GetSpacing())
		out_img.SetOrigin(self.img.GetOrigin())
		out_img.SetDirection(self.img.GetDirection())
		sitk.writeImage(out_img,output_path)

	def getCurrAndMaxIdx(self):
		return 0, self.vol_data.shape[0]

	def loadAndProcessVolume(self,vol_path):
		print("loading the image...")
		self.img = self.loadImage(vol_path)
		self.vol_data = sitk.GetArrayFromImage(self.img)
		
		self.out_data = np.zeros(self.vol_data.shape)
		self.posteriors = np.zeros(self.vol_data.shape[0],self.vol_data.shape[1],self.vol_data.shape[2],self.num_class)
		self.curr_idx, self.max_idx = self.getCurrAndMaxIdx()

		print("windowing ... ")
		self.vol_data1, self.vol_data2, self.vol_data3 = self.HU_window(self.vol_data)
		print("equalizing the histograms ...")
		self.vol_data1 = self.histogram_equalization(self.vol_data1)
		self.vol_data2 = self.histogram_equalization(self.vol_data2)
		self.vol_data3 = self.histogram_equalization(self.vol_data3)
		print("normalizing the values ...")
		self.vol_data1 = self.normalize(self.vol_data1)
		self.vol_data2 = self.normalize(self.vol_data2)
		self.vol_data3 = self.normalize(self.vol_data3)

		print("downsampling the image ...")
		self.vol_data1 = self.downSample(self.vol_data1)
		self.vol_data2 = self.downSample(self.vol_data2)
		self.vol_data3 = self.downSample(self.vol_data3)

	def getLargestConnectedComponent(self,vol_data):
		data = vol_data.copy()
		data[data == 2] = 1
		c,n = snd.label(data)
		sizes = snd.sum(data, c, range(n+1))
		mask_size = sizes < (max(sizes))
		remove_voxels = mask_size[c]
		vol_data[remove_voxels] = 0
		# c[np.where(c!=0)]= 1

		return vol_data

	def binary_dilation(self,data):
		image = sitk.GetImageFromArray(data)
		out = self.dilation_filter.Execute(image)
		return sitk.GetArrayFromImage(out)

	def binary_erosion(self,data):
		image = sitk.GetImageFromArray(data)
		out = self.erosion_filter.Execute(image)
		return sitk.GetArrayFromImage(out)		

	def runForwardOnBatch(self, net,input_batch):
		outputs, posteriors = self.sess.run([net.predictions,net.posteriors], feed_dict={net.inputs:input_batch,net.is_training:False})

		return outputs[0], posteriors[0]

	def prediction_iterator(self):
		from_idx = self.curr_idx
		
		while from_idx < self.vol_data.shape[0]
			to_idx = from_idx + self.batch_size
			start_time = time.time()
			if to_idx < self.max_idx:
				inputs_shape = (self.batch_size,self.vol_data1.shape[1],self.vol_data1.shape[2],self.num_channels)
				input_batch = np.zeros(inputs_shape)
				input_batch[:,:,:,0] = self.vol_data1[from_idx:to_idx]
				input_batch[:,:,:,1] = self.vol_data2[from_idx:to_idx]
				input_batch[:,:,:,2] = self.vol_data3[from_idx:to_idx]
			else:
				to_idx = max_idx
				inputs_shape = (self.max_idx - from_idx, self.vol_data1.shape[1], self.vol_data1.shape[2],self.num_channels)
				input_batch = np.zeros(inputs_shape)
				input_batch[:,:,:,0] = self.vol_data1[from_idx:to_idx]
				input_batch[:,:,:,1] = self.vol_data2[from_idx:to_idx]
				input_batch[:,:,:,2] = self.vol_data3[from_idx:to_idx]

			outputs, posteriors = self.runForwardOnBatch(self.net, self.input_batch)

			self.out_data[from_idx:to_idx] = outputs
			self.posteriors[from_idx:to_idx] = posteriors

			utils.Progress(from_idx,self.max_idx,time.time()-start_time)

			from_idx += self.batch_size
	
	def postProcessOutputs(self):
		print("Post processing the network outputs ... ")
		self.out_data = self.binary_erosion(self.out_data)
		self.out_data = self.getLargestConnectedComponent(self.out_data)
		self.out_data = self.binary_dilation(self.out_data)

		self.out_data = np.uint8(self.out_data)

	def runPrediction(self):
		liver_dice_mat = []
		tumor_dice_mat = []
		for each_tuple in self.vols:
			vol_path = each_tuple[0]
			seg_path = self.segs[self.vols.index(each_tuple)]
			save_path = each_tuple[1]
			vol_name = vol_path.split('/')[-1]
			posteriors_save_dir = os.path.join(self.output_folder_path,'posteriors')
			posteriors_save_path = os.path.join(posteriors_save_dir,vol_name)

			if not os.path.exists(posteriors_save_dir):
				os.makedirs(posteriors_save_dir)

			print("working on " + vol_name)
			self.loadAndProcessVolume(vol_path)
			self.prediction_iterator()
			self.postProcessOutputs()

			print("Calculating dice ...")
			liver_dice, tumor_dice = self.calculateDice(self.out_data)
			print(vol_name + " - Liver dice : " + str(liver_dice) + " | Tumor dice : " + str(tumor_dice))
			
			liver_dice_mat.append(liver_dice)
			tumor_dice_mat.append(tumor_dice)
			
			self.writeDicomImage(self.out_data, save_path)
			self.writeDicomImage(self.posteriors, posteriors_save_path)

		print("Average Liver Dice : "+ str(np.mean(liver_dice_mat)))
		print("Average Tumor Dice : "+ str(np.mean(tumor_dice_mat)))

if __name__ == __main__():
	data_folder_path = '/media/brats/Varghese/lits2017/test/'
	output_folder_path = '/media/brats/Varghese/lits2017/predictions/tiramisu_2d/'
	model_path = '/media/brats/Varghese/lits2017/outputs/tiramisu_liver_tumor_edge_weights_3channel_512_out/best_model/latest.ckpt'
	batch_size = 4
	num_channels = 3
	num_class = 3
	gpu_ids = [0]

	predictor = Predictor(data_folder_path,output_folder_path,model_path,batch_size,num_class,num_channels,gpu_ids)

	predictor.runPrediction()