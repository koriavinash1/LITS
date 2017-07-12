import SimpleITK as sitk
import numpy as np

img_src_path = "/media/brats/Varghese/lits2017/train/"
img_des_path = "/media/brats/Varghese/lits2017/NewDataSet/train/"

def get_all_volumes_paths(base_path):
    return [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.nii')]

def get_input_paths(paths):
    return [path for path in paths if path.split("/").pop().split(".")[0].split("_")[0].split("-")[0] == "volume"]

def get_segmentation_paths(paths):
    return [path for path in paths if path.split("/").pop().split(".")[0].split("_")[0].split("-")[0] == "segmentation"]

def extractROI(volume, segmentation):
    buffer_slice = 5
    roi = volume*segmentation
    x, y, z = np.where(roi!=0)
    min_coords = (np.min(x)+buffer_slice, np.min(y)+buffer_slice, np.min(z)+buffer_slice)
    max_coords = (np.max(x)+buffer_slice, np.max(y)+buffer_slice, np.max(z)+buffer_slice)
    return min_coords, max_coords

def extractPatchs(volume, segmentation,
                                min_coords, max_coords):
    patchSize = 120
    pathTemplate = np.zeros(patchSize,patchSize,patchSize)
    for i in range((max_coords[2]-min_coords[2])*3/patchSize):
        for j in range((max_coords[1]-min_coords[1])*3/patchSize):
            for k in range((max_coords[0]-min_coords[0])*3/patchSize):
