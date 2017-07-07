import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import time

img_src_path = "../../DataSet/Training Batch 1/"
img_des_path = "../../DataSet/NewDataSet/"
if not os.path.exists(img_des_path):
    os.mkdir(img_des_path)

csv_path =  "../../DataSet/DataInfo.csv"
fileHeaders = "OriginalImage, OriginalSize, OriginalSpacing"

labelFile = open(csv_path, 'w')
labelFile.write(fileHeaders)
labelFile.close()

def loadAconvert_nii_files(img_path, des_path):
	name = img_path.split("/").pop().split(".")[0]
	itk_image = sitk.ReadImage(img_path)
	img_array = sitk.GetArrayFromImage(itk_image)
    	original_size, original_spacing = itk_image.GetSize(), itk_image.GetSpacing()
    	
    	new_size0, new_spacing0 = (32, 32, 32), (1,1,1)
    	new_size1, new_spacing1 = (64, 64, 64), (1,1,1)
    	new_size2, new_spacing2 = (128, 128, 128), (1,1,1)

    	resampled_itk_img0 = sitk.Resample(itk_image, new_size0, sitk.Transform(), 
                                  sitk.sitkNearestNeighbor, image.GetOrigin(),
                                  new_spacing0, image.GetDirection(), 0.0, 
                                  itk_image.GetPixelID())
   	resampled_itk_img1 = sitk.Resample(itk_image, new_size1, sitk.Transform(), 
                                  sitk.sitkNearestNeighbor, image.GetOrigin(),
                                  new_spacing1, image.GetDirection(), 0.0, 
                                  itk_image.GetPixelID())
   	resampled_itk_img2= sitk.Resample(itk_image, new_size2, sitk.Transform(), 
                                  sitk.sitkNearestNeighbor, image.GetOrigin(),
                                  new_spacing2, image.GetDirection(), 0.0, 
                                  itk_image.GetPixelID())
   	
   	np.save(des_path+name+"_size_32.npy", 
   		np.clip(GetArrayFromImage(resampled_itk_img0), -100, 400))
   	np.save(des_path+name+"_size_64.npy", 
   		np.clip(GetArrayFromImage(resampled_itk_img1), -100, 400))
   	np.save(des_path+name+"_size_128.npy", 
   		np.clip(GetArrayFromImage(resampled_itk_img2), -100, 400))
   	
   	return original_size, original_spacing
 	
def write_to_csv(csv_path, info):
	labelFile = open(csv_path, 'a')
	labelFile.write("\n")
	labelFile.write(info)
	labelFile.close()
	pass

def generate_new_data(base_src_path, base_des_path):
	img_paths = [os.path.join(base_src_path, f) for f in os.listdir(base_src_path) if f.endswith('.nii')]
	for img_path in img_paths:
		print "converting image: {}".format(img_path)
		original_size, original_spacing = loadAconvert_nii_files(img_path, base_des_path)
		write_to_csv(csv_path, img_path+","+str(original_size)+","+str(original_spacing))
	pass

if __name__ == "__main__":
	start = time.time()
	generate_new_data(img_src_path, img_des_path)
	print "Total time taken for convertion: {}sec".format(time.time()-start)
	print "All the volumes got converted to npy files, location: {}".format(img_des_path)
	print "If any problems rise github issues"
