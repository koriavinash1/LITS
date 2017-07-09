import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import time
import matplotlib.pyplot as plt
import numbers
from nilearn.image import resample_img
from scipy.ndimage.interpolation import zoom
from skimage.transform import resize
from skimage.morphology import dilation

img_src_path = "../../DataSet/train/"
img_des_path = "../../DataSet/NewDataSet/train/"
if not os.path.exists(img_des_path):
    os.mkdir(img_des_path)

csv_path =  "../../DataSet/DataInfo.csv"
fileHeaders = "OriginalImage, OriginalAffine, OriginalHeader"

labelFile = open(csv_path, 'w')
labelFile.write(fileHeaders)
labelFile.close()

def resample_resize_nii(target_spacing, target_shape, 
						img_array, img_affine, img_hdr):
	if isinstance(target_spacing, numbers.Number):
	    target_spacing = [target_spacing] * img_array.ndim
	print 'Before processing'
	print img_array.shape
	print 'After processing'
	# compute zoom values
	# dimensionality = sum(0 if 1 == x else 1 for x in img_array.shape)
	# zoom_factors = [old / float(new) for new, old in zip(target_spacing, img_hdr.get_zooms()[0:dimensionality])]
	# img_array = zoom(img_array, zoom_factors, order=0, mode='constant')

	if not len(img_hdr.get_zooms()) == len(target_spacing):
	  	raise AttributeError('Vector dimensions of header ({}) and supplied spacing sequence ({}) differ.'.format(len(img_hdr.get_zooms()), len(target_spacing)))
	img_hdr.set_zooms([float(x) for x in target_spacing])  

	target_affine = np.eye(4)
	target_affine[0,0] = -1*target_spacing[0]
	target_affine[1,1] = -1*target_spacing[1]
	target_affine[2,2] = 1*target_spacing[2]

	# Save the resampled image in .nii in output_path specified
	resized_vol = resize(img_array, target_shape, mode='reflect')
	dilated_vol = dilation(resized_vol)
	print dilated_vol.shape
	nimg = nib.Nifti1Image(resized_vol, affine=img_affine, header=img_hdr)
	nimg = resample_img(nimg, target_affine=img_affine, target_shape=target_shape,
	interpolation='nearest', copy=True, order='F')
	return nimg

def load_and_convert_nii(img_path, des_path):
	name = img_path.split("/").pop().split(".")[0]
	nimg = nib.load(img_path)
	img_array, img_affine, img_hdr = nimg.get_data(), nimg.affine, nimg.header
	target_spacing, target_shape = [1, 1, 1], [32,32,32]
	nimg = resample_resize_nii(target_spacing, target_shape, 
						img_array, img_affine, img_hdr)
	nimg.to_filename(des_path+name+'_size_64.nii')   	
   	return img_affine, img_hdr
 	
def write_to_csv(csv_path, info):
	labelFile = open(csv_path, 'a')
	labelFile.write("\n")
	labelFile.write(info)
	labelFile.close()
	pass

def generate_new_data(base_src_path, base_des_path):
	img_paths = [os.path.join(base_src_path, f) for f in os.listdir(base_src_path) if f.endswith('.nii')]
	print "Total number of images :{}".format(len(img_paths))
	for img_path in img_paths:
		print "converting image: {}".format(img_path)
		original_affine, original_hdr = load_and_convert_nii(img_path, base_des_path)
		write_to_csv(csv_path, img_path+","+str(original_affine)+","+str(original_hdr))
	pass

if __name__ == "__main__":
	start = time.time()
	generate_new_data(img_src_path, img_des_path)
	print "Total time taken for convertion: {}sec".format(time.time()-start)
	print "All the volumes got converted to npy files, location: {}".format(img_des_path)
	print "If any problems rise github issues"
