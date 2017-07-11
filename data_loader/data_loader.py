from __future__ import division
import numpy as np
import os, sys, shutil
import h5py
import random, time
import scipy.ndimage as snd
import skimage.morphology as morph
import weakref
import threading
import random
sys.path.insert(0,'../utils/')
import train_utils as utils

def worker(weak_self):
	self = weak_self()
	name  = threading.current_thread().name

	while not self.done_event.is_set():
		if (self.iter_over == False) and (self.n_imgs_in_ram < self.max_imgs_in_ram):
			with self.file_access_lock:
				input_path = self.popFilePath()

				if (input_path is not None) and (input_path not in self.files_accessed):
					self.files_accessed.append(input_path)
					image, label, weight = self.getDataFromPath(input_path)

					with self.data_access_lock:
						if self.image_volume.size != 0  and self.label_volume.size != 0:
							try:
								self.image_volume = np.vstack([self.image_volume, image])
								self.label_volume = np.vstack([self.label_volume, label])
								self.weight_volume = np.vstack([self.weight_volume, weight])
							except Exception as e:
								print(str(e))
								self.image_volume = np.array([])
								self.label_volume = np.array([])
								self.weight_volume = np.array([])
								print('Image queue shape: ' + str(self.image_volume.shape))
								print('Image slice shape: ' + str(image.shape))
								print('Label queue shape: ' + str(self.label_volume.shape))
								print('Label slice shape: ' + str(label.shape))
						else:
							self.image_volume = image
							self.label_volume = label
							self.weight_volume = weight

						self.n_imgs_in_ram = self.image_volume.shape[0]
				
				elif input_path == None:
					self.iter_over_for_thread[name] = True



class ITERATOR(object):
	def __init__(self, data_folder_path, mode='train', batch_size = 2, num_threads = 4, max_imgs_in_ram = 500):
		self.data_folder_path = data_folder_path
		self.mode = mode
		self.batch_size = batch_size
		self.num_threads = num_threads
		self.iter_over = False
		self.mode = 'train'
		self.image_volume = np.array([])
		self.label_volume = np.array([])
		self.weight_volume = np.array([])
		self.n_imgs_in_ram = 0
		self.num_imgs_obt = 0
		self.max_imgs_in_ram = max_imgs_in_ram
		self.files_accessed = []
		self.file_access_lock = threading.Lock()
		self.data_access_lock = threading.Lock()
		self.done_event = threading.Event()
		self.iter_over_for_thread = {}

		self.getFilePaths(data_folder_path)

		for t_i in range(0,num_threads):
			t = threading.Thread(target = worker,args = (weakref.ref(self),))
			t.setDaemon(True)
			t.start()
			self.iter_over_for_thread[t.name] = False

	def getFilePaths(self,data_folder_path):
		self.train_fls = [os.path.join(data_folder_path,'train',f) for f in os.listdir(os.path.join(data_folder_path,'train'))]
		self.val_fls = [os.path.join(data_folder_path,'val',f) for f in os.listdir(os.path.join(data_folder_path,'val'))]
		random.shuffle(self.train_fls)
		random.shuffle(self.val_fls)

	def popFilePath(self):
		if self.mode == 'train':
			if len(self.train_fls) > 0:
				return self.train_fls.pop()
			else:
				return None
		elif self.mode == 'val':
			if len(self.val_fls) > 0:
				return self.val_fls.pop()
			else:
				return None
		else:
			print("Got unknown value for mode, supported values are : 'train', 'val' ")
			raise 1

	def getDataFromPath(self,path):
		h5 = h5py.File(path,'r')
		img = h5['image'][:]
		lbl = h5['label'][:]
		weight = h5['weight_map'][:]

		return img, lbl, weight

	def getNextBatch(self):
		while True:
			with self.data_access_lock:
				image_batch,self.image_volume = np.split(self.image_volume,[self.batch_size])
				label_batch,self.label_volume = np.split(self.label_volume,[self.batch_size])
				weight_batch,self.weight_volume = np.split(self.weight_volume,[self.batch_size])

				num_imgs_obt = image_batch.shape[0]
				self.n_imgs_in_ram = self.image_volume.shape[0]

			if ((sum(x == True for x in self.iter_over_for_thread.values()) == self.num_threads) and (self.n_imgs_in_ram == 0)):
				self.iter_over = True

			if (num_imgs_obt > 0) or self.iter_over :
				if (num_imgs_obt != self.batch_size) and temp_count <=3 :
					time.sleep(2)
					temp_count += 1
				else:
					break

		return image_batch, label_batch, weight_batch

	def reset(self):
		self.image_volume = np.array([])
		self.label_volume = np.array([])
		self.weight_volume = np.array([])
		self.n_imgs_in_ram = self.image_volume.shape[0]
		self.train_fls = []
		self.val_fls = []
		self.getFilePaths(self.data_folder_path)
		self.files_accessed = []
		for key in self.iter_over_for_thread:
			self.iter_over_for_thread[key] = False	
		self.iter_over = False

	def __del__(self):
		print(' Thread exited ')
		self.done_event.set()
