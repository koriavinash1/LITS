import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import time
import matplotlib.pyplot as plt
import SimpleITK as sitk


img_src_path = "/media/brats/Varghese/lits2017/train/"
img_des_path = "/media/brats/Varghese/lits2017/NewDataSet/train/"

if not os.path.exists(img_des_path):
    # os.mkdir("/media/brats/Varghese/lits2017/NewDataSet/")
    os.mkdir(img_des_path)

csv_path =  "/media/brats/Varghese/lits2017/DataInfo.csv"
fileHeaders = "OriginalImage, OriginalAffine, OriginalHeader"

labelFile = open(csv_path, 'w')
labelFile.write(fileHeaders)
labelFile.close()


def resample(image,interpolator=sitk.sitkLinear, shape=(64,64,64)):
	inputSize = image.GetSize()
	inputSpacing = image.GetSpacing()
	print "Input Spacing {}".format(inputSpacing)
	# For making Isotropic volume
	# outputSpacing = (1.0, 1.0, 1.0)
	# outputSize[0] = int(inputSize[0] * inputSpacing[0] / outputSpacing[0] + .5)
	# outputSize[1] = int(inputSize[1] * inputSpacing[1] / outputSpacing[1] + .5)
	# outputSize[2] = int(inputSize[2] * inputSpacing[2] / outputSpacing[2] + .5)
	outputSize = shape
	print "Input Shape {}".format(sitk.GetArrayFromImage(image).shape)
	temp = (inputSize[0]/float(outputSize[0]),inputSize[1]/float(outputSize[1]),inputSize[2]/float(outputSize[2]))
	outputSpacing = (inputSpacing[0]*temp[0],inputSpacing[1]*temp[1],inputSpacing[2]*temp[2])
	print "Output Spacing {}".format(outputSpacing)
	outputSize = tuple(outputSize)
	resampler = sitk.ResampleImageFilter()
	resampler.SetSize(outputSize)
	resampler.SetOutputSpacing(outputSpacing)
	resampler.SetOutputOrigin(image.GetOrigin())
	resampler.SetOutputDirection(image.GetDirection())
	resampler.SetInterpolator(interpolator)
	resampler.SetDefaultPixelValue(999.0)
	image = resampler.Execute(image)
	print "Output Shape {}".format(sitk.GetArrayFromImage(image).shape)
	return image

def ContrastAdjust(nimg,contrastRange=(-100, 400)):
	npimage = nimg.get_data()
	nimg_affine = nimg.get_affine()
	img_min = np.min(npimage)
	img_max = np.max(npimage)
	print img_min, img_max
	max_value = contrastRange[1]
	min_value = contrastRange[0]
	# TODO: Think
	processedimage = np.round((((npimage-img_min)/np.float32((img_max-img_min)))*np.float32((max_value-min_value))) + min_value).astype("int32")

	nimg = nib.Nifti1Image(processedimage, nimg_affine)
	return nimg
	
def load_and_convert_nii(img_path, des_path):
	name = img_path.split("/").pop().split(".")[0]
	nimg = sitk.ReadImage(img_path)

	target_shape =  (128,128,128)
	if name.split("-")[0] == "volume":
		img_obj = resample(nimg, sitk.sitkLinear, shape=target_shape)
	else:
		img_obj = resample(nimg, sitk.sitkNearestNeighbor,shape=target_shape)
	sitk.WriteImage(img_obj, des_path+name+'_size_128.nii')
   	return nimg.GetDirection(), nimg.GetOrigin(), nimg.GetPixelID()
 	
def write_to_csv(csv_path, info):
	labelFile = open(csv_path, 'a')
	labelFile.write("\n")
	labelFile.write(info)
	labelFile.close()
	pass

def generate_new_data(base_src_path, base_des_path):
	img_paths = [os.path.join(base_src_path, f) for f in os.listdir(base_src_path) if f.endswith('.nii')]
	# img_paths=[base_src_path+"/"+"volume-109.nii", base_src_path+"/"+"segmentation-109.nii"]
	print "Total number of images :{}".format(len(img_paths))
	for img_path in img_paths:
		print "converting image: {}".format(img_path)
		direction_info, origin_info, pixelID = load_and_convert_nii(img_path, base_des_path)
		write_to_csv(csv_path, img_path+","+str(direction_info)+","+str(origin_info)+","+str(pixelID))
	pass

if __name__ == "__main__":
	start = time.time()
	generate_new_data(img_src_path, img_des_path)
	print "Total time taken for convertion: {}sec".format(time.time()-start)
	print "All the volumes got converted to npy files, location: {}".format(img_des_path)
	print "If any problems rise github issues"