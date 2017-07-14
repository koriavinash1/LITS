import glob
import re
import itertools
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import cv2
# Custom
import sys
sys.path.append("../")
from data_preprocess import utils
from aug import roi_patch_transform_norm, roi_patch_norm, sample_augmentation_parameters


def channel_grouping(data_tuple, axis=1):
    """
    The shape of data is H*W*groups*slices
    returns slices*H*W*Group
    """
    nchannels = len(data_tuple)
    slice_n = data_tuple[0].shape[0]
    if axis:
        shape = (slice_n,)+data_tuple[0].shape[1:]+(nchannels,)
    else:
        shape = (slice_n,)+(nchannels,)+data_tuple[0].shape[1:]
    # print data_tuple[0].dtype, data_tuple[1].dtype, data_tuple[2].dtype
    data_batch = np.zeros(shape, dtype='float32')
    for i in range(slice_n):
        if axis:
            data_batch[i] = np.dstack([dp[i] for dp in data_tuple])
        else:
            data_batch[i] = np.swapaxes(np.swapaxes(np.dstack([dp[i] for dp in data_tuple]),0,2),1,2)
    return  data_batch

def count_total_slices(pickled_data):
    """
    Count the total number of images
    in the pickled dataset comprising of ES and ED volumes
    """
    cnt=0
    for pid in pickled_data.keys():
        # Volume shape: H*W*Slices
        # Count slices
        # Data comprises of both ED and ES -> twice
        cnt+=pickled_data[pid]['ED_GT'].shape[-1]*2
    return cnt


class PatientwiseSliceGenerator(object):
    """
    Depending on batch_size number of patient volumes are extracted
    """
    def __init__(self, pickle_data_path, transform_params, nlabels, patient_batch_size=1, batch_size=10, nchannels=1, random=False, aug=False,
                infinite=False, shuffle_patient_batch=False, data_prep_fun=roi_patch_transform_norm,
                test_data_prep_fun=roi_patch_norm, test_mask=False, fixed_batch_size=True, **kwargs):

        self.data = utils.load_pkl(pickle_data_path)
        self.npatients = len(self.data.keys())
        self.batch_size = batch_size
        self.rng = np.random.RandomState(40)
        self.nlabels = nlabels
        self.nchannels = nchannels
        self.random = random
        self.aug = aug
        self.infinite = infinite
        self.transformation_params = transform_params
        self.data_prep_fun = data_prep_fun
        self.test_data_prep_fun = test_data_prep_fun
        self.patient_batch_size = patient_batch_size
        self.nsamples = count_total_slices(self.data)
        self.shuffle_patient_batch = shuffle_patient_batch
        self.test_mask = test_mask
        self.fixed_batch_size = fixed_batch_size

    def get_n_classes(self):
        return self.nlabels

    def get_n_samples(self):
        return self.nsamples

    def get_n_batches(self):
        return (self.nsamples/self.batch_size)

    def get_n_patient_batches(self):
        return int(np.ceil(self.npatients/self.patient_batch_size))

    def get_n_patients(self):
        return self.npatients        

    def get_void_labels(self):
        """returns a list containing the classes associated to void"""
        # TODO: Better Handling of void classes: workaround - return non-existing class label
        return [5]

    def test_generator(self):
        patients_ids = sorted(self.data.keys())
        for pid in patients_ids:
            img_size = self.data[pid]['4D'].shape[:2]
            affine = self.data[pid]['nib']['affine_gt']
            hdr = self.data[pid]['nib']['hdr_gt']
            roi_center = self.data[pid]['roi_center']
            x_batch_shape = (0,)+img_size+(self.nchannels,)
            y_batch_shape = (0,)+img_size
            x_batch = np.empty(x_batch_shape, dtype='float32')
            y_batch = np.empty(y_batch_shape, dtype='uint8')
            minibatch_data = self.test_data_prep_fun(self.data[pid], self.test_mask)
            # ED
            x_ed_img_batch = np.swapaxes(np.swapaxes(minibatch_data['ED'],0,2),1,2)
            y_ed_gt_batch = np.swapaxes(np.swapaxes(minibatch_data['ED_GT'],0,2),1,2)
            # ES
            x_es_img_batch = np.swapaxes(np.swapaxes(minibatch_data['ES'],0,2),1,2)
            y_es_gt_batch = np.swapaxes(np.swapaxes(minibatch_data['ES_GT'],0,2),1,2)
            roi_mask = np.swapaxes(np.swapaxes(minibatch_data['ROI'],0,2),1,2)
            if self.nchannels == 1:
                x_img_batch = np.expand_dims(np.concatenate((x_ed_img_batch, x_es_img_batch), axis=0), axis=3)
            if self.nchannels == 2:
                # FFT_0, FFT_1
                x_fft1_img_batch = np.swapaxes(np.swapaxes(minibatch_data['FFT_1'],0,2),1,2)
                x_ed_img_batch = channel_grouping((x_ed_img_batch, x_fft1_img_batch), axis=3)
                x_es_img_batch = channel_grouping((x_es_img_batch, x_fft1_img_batch), axis=3)
                x_img_batch = np.concatenate((x_ed_img_batch, x_es_img_batch), axis=0)
            if self.nchannels == 3:
                # FFT_0, FFT_1
                x_fft0_img_batch = np.swapaxes(np.swapaxes( minibatch_data['FFT_0'],0,2),1,2)
                x_fft1_img_batch = np.swapaxes(np.swapaxes(minibatch_data['FFT_1'],0,2),1,2)
                x_ed_img_batch = channel_grouping((x_ed_img_batch, x_fft0_img_batch, x_fft1_img_batch), axis=3)
                x_es_img_batch = channel_grouping((x_es_img_batch, x_fft0_img_batch, x_fft1_img_batch), axis=3)
                x_img_batch = np.concatenate((x_ed_img_batch, x_es_img_batch), axis=0)
            y_gt_batch = np.concatenate((y_ed_gt_batch, y_es_gt_batch), axis=0)
            x_batch = np.append(x_batch, x_img_batch, axis=0)
            y_batch = np.append(y_batch, y_gt_batch, axis=0)
            yield x_batch, y_batch, pid, affine, hdr, roi_mask, roi_center

    def __call__(self):
        while True:
            patients_ids = sorted(self.data.keys())
            if self.random:
                self.rng.shuffle(patients_ids)

            patch_size = self.transformation_params['patch_size']
            x_batch_shape = (0,)+patch_size+(self.nchannels,)
            y_batch_shape = (0,)+patch_size
            for idx in range(self.get_n_patient_batches()):
                x_batch = np.empty(x_batch_shape, dtype='float32')
                y_batch = np.empty(y_batch_shape, dtype='uint8')
                pid_list = []
                for _id in range(self.patient_batch_size):
                    tracker = min(idx*self.patient_batch_size + _id, self.npatients)
                    # sample augmentation params per patient
                    random_params = None
                    if self.aug:
                        random_params = sample_augmentation_parameters(self.transformation_params)
                        # print random_params
                    pid = patients_ids[tracker]
                    minibatch_data = self.data_prep_fun(self.data[pid], self.transformation_params, nlabel=self.nlabels,
                                        random_augmentation_params=random_params, uniform_scale=True if random_params else False)
                    # ED
                    x_ed_img_batch = np.swapaxes(np.swapaxes( minibatch_data['ED'],0,2),1,2)
                    y_ed_gt_batch = np.swapaxes(np.swapaxes(minibatch_data['ED_GT'],0,2),1,2)
                    # ES
                    x_es_img_batch = np.swapaxes(np.swapaxes(minibatch_data['ES'],0,2),1,2)
                    y_es_gt_batch = np.swapaxes(np.swapaxes(minibatch_data['ES_GT'],0,2),1,2)

                    if self.nchannels == 1:
                        x_img_batch = np.expand_dims(np.concatenate((x_ed_img_batch, x_es_img_batch), axis=0), axis=3)
                    if self.nchannels == 2:
                        # FFT_0, FFT_1
                        x_fft1_img_batch = np.swapaxes(np.swapaxes(minibatch_data['FFT_1'],0,2),1,2)
                        x_ed_img_batch = channel_grouping((x_ed_img_batch, x_fft1_img_batch), axis=3)
                        x_es_img_batch = channel_grouping((x_es_img_batch, x_fft1_img_batch), axis=3)
                        x_img_batch = np.concatenate((x_ed_img_batch, x_es_img_batch), axis=0)
                    if self.nchannels == 3:
                        # FFT_0, FFT_1
                        x_fft0_img_batch = np.swapaxes(np.swapaxes( minibatch_data['FFT_0'],0,2),1,2)
                        x_fft1_img_batch = np.swapaxes(np.swapaxes(minibatch_data['FFT_1'],0,2),1,2)
                        x_ed_img_batch = channel_grouping((x_ed_img_batch, x_fft0_img_batch, x_fft1_img_batch), axis=3)
                        x_es_img_batch = channel_grouping((x_es_img_batch, x_fft0_img_batch, x_fft1_img_batch), axis=3)
                        x_img_batch = np.concatenate((x_ed_img_batch, x_es_img_batch), axis=0)

                    y_gt_batch = np.concatenate((y_ed_gt_batch, y_es_gt_batch), axis=0)
                    x_batch = np.append(x_batch, x_img_batch, axis=0)
                    y_batch = np.append(y_batch, y_gt_batch, axis=0)
                    pid_list.append(pid)
                    # Shuffle before yielding
                    if self.shuffle_patient_batch:
                        temp = np.c_[x_batch.reshape(len(x_batch), -1), y_batch.reshape(len(y_batch), -1)]
                        self.rng.shuffle(temp)
                        x_batch = temp[:, :x_batch.size//len(x_batch)].reshape(x_batch.shape)
                        y_batch = temp[:, x_batch.size//len(x_batch):].reshape(y_batch.shape).astype('uint8')
                mini_pat_batch = x_batch.shape[0]

                if not self.fixed_batch_size:
                    for i in range(0, mini_pat_batch, self.batch_size):
                        yield x_batch[i:min(i + self.batch_size, mini_pat_batch)], y_batch[i:min(i + self.batch_size, mini_pat_batch)], pid_list
                else:
                    # TODO: code to get fixed sized batches
                    # Will not miss any images in the entire patient minibatch if np.ceil function is used
                    n_mini_batches = int(np.ceil(mini_pat_batch/float(self.batch_size)))
                    for i in range(0, n_mini_batches):
                        iloc = i*self.batch_size
                        if iloc+self.batch_size >= mini_pat_batch:
                            iloc = (mini_pat_batch - self.batch_size) 
                        # print i, iloc, mini_pat_batch, n_mini_batches    
                        yield x_batch[iloc:(iloc + self.batch_size)], y_batch[iloc:(iloc + self.batch_size)], pid_list                    
            if not self.infinite:
                break

def load_data(batch_size=10, patient_batch_size=4):
    # Pickle file path
    fulldata_pickled_path = '../../../../dataset/processed_data/pickled/complete_patient_data.pkl'
    train_pickled_path = '../../../../dataset/processed_data/pickled/train_patient_data.pkl'
    validation_pickled_path = '../../../../dataset/processed_data/pickled/validation_patient_data.pkl'
    test_pickled_path = '../../../../dataset/processed_data/pickled/test_patient_data.pkl'
    # Set patch extraction parameters
    size1 = (256, 256)
    size2 = (128, 128)
    size3 = (200, 200)
    size4 = (160, 160)   
    patch_size = size2
    mm_patch_size = size2
    train_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
        # 'rotation_range': (-15, 45),
        # 'translation_range_x': (-3, 3),
        # 'translation_range_y': (-3, 3),
        # 'zoom_range': (0.8, 1.2),
        # 'do_flip': (False, True),
    }
    valid_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
    }
    test_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
    }
    n_labels = 4
    train_iter = PatientwiseSliceGenerator(train_pickled_path, train_transformation_params, n_labels, patient_batch_size=patient_batch_size, batch_size=batch_size, nchannels=1,
                    random=False, aug=False, infinite=False, shuffle_patient_batch=False, data_prep_fun=roi_patch_transform_norm, fixed_batch_size=True)
    valid_iter = PatientwiseSliceGenerator(validation_pickled_path, valid_transformation_params, n_labels, patient_batch_size=patient_batch_size, batch_size=batch_size, nchannels=1,
                    random=False, aug=False, infinite=False, shuffle_patient_batch=False, data_prep_fun=roi_patch_transform_norm, fixed_batch_size=True)
    test_iter = PatientwiseSliceGenerator(test_pickled_path, test_transformation_params, n_labels, patient_batch_size=patient_batch_size, batch_size=batch_size, nchannels=1,
                    random=False, aug=False, infinite=False, shuffle_patient_batch=False, data_prep_fun=roi_patch_transform_norm,
                    test_data_prep_fun=roi_patch_norm, test_mask=False, fixed_batch_size=True)

    return train_iter, valid_iter, test_iter

def load_data_full_test():
    # Pickle file path
    fulldata_pickled_path = '../../../../dataset/processed_data/pickled/complete_patient_data.pkl'
    test_pickled_path = '../../../../dataset/processed_data/pickled/test_patient_data.pkl'
    # Set patch extraction parameters
    size1 = (256, 256)
    size2 = (128, 128)
    size3 = (200, 200)
    size4 = (160, 160)   
    patch_size = size2
    mm_patch_size = size2
    test_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
    }
    n_labels = 4
    test_iter = PatientwiseSliceGenerator(test_pickled_path, test_transformation_params, n_labels, patient_batch_size=1, batch_size=1, nchannels=1,
                    random=False, aug=False, infinite=False, shuffle_patient_batch=False, data_prep_fun=roi_patch_transform_norm,
                    test_data_prep_fun=roi_patch_norm, test_mask=False, fixed_batch_size=False)
    return test_iter

def extract_patch(image_4D, roi_center, patch_size=128):
    """
    This code extracts a patch of defined size from the given center point of the image
    and returns parameters for padding and translation to original location of the image
    Args:
    4D: Volume: Batch_size, X ,Y, channels
    """
    rows, cols = image_4D.shape[1:3]
    patch_rows = max(0, roi_center[0] - patch_size/2)
    patch_cols = max(0, roi_center[1] - patch_size/2)
    patch_img = image_4D[:, patch_rows:patch_rows+patch_size, patch_cols:patch_cols+patch_size,:]
    # patch_img = np.expand_dims(patch_img, axis=1)
    pad_params = {'rows': rows, 'cols': cols, 'tx': patch_cols, 'ty': patch_rows}
    return patch_img, pad_params

def pad_patch(patch_3D, pad_params):
    """
    This code does padding and translation to original location of the image
    Args:
    3D: Volume: Batch_size, X, Y
    Used for predicted ground truth
    """
    if patch_3D.dtype != 'float32':
        dtype = 'uint8'
    else:
        dtype = 'float32'
    patch_3D = patch_3D.astype(dtype)
    M = np.float32([[1,0, pad_params['tx']],[0, 1, pad_params['ty']]])
    padded_patch = np.empty(((0,)+(pad_params['rows'], pad_params['cols'])), dtype=np.float32)

    for i in range(patch_3D.shape[0]):
        # import pdb; pdb.set_trace()
        patch = cv2.warpAffine(patch_3D[i],M,(pad_params['cols'], pad_params['rows']))
        patch = np.expand_dims(patch, axis=0)
        padded_patch = np.append(padded_patch, patch, axis=0)
    return padded_patch.astype(dtype)


if __name__ == "__main__":
    # train_iter, valid_iter, test_iter = load_data()
    # # for X, Y, pid  in train_iter():
    # #     print pid
    # p = train_iter()
    # for X, Y, pid  in p:
    #     print pid
    # print 'Hi'
    # for X, Y, pid  in p:
    #     print pid
        # print X.shape, X.dtype
        # print Y.shape, Y.dtype
        # for j in range(X.shape[0]):
        #     utils.imshow(X[j,:,:,0].T, Y[j,:,:].T)
    iterator = load_data_full_test()
    _iter = iterator.test_generator()
    n_batches = iterator.get_n_patients()
    for i in range(n_batches):
        X, Y, pid, affine, hdr, roi_mask, roi_center  = _iter.next()
        if pid == 'patient057':
            print pid
            print X.shape
            # print X.shape, X.dtype
            # print Y.shape, Y.dtype
    #         print roi_center
    #         print affine
    #         patch_img, pad_params = extract_patch(X, roi_center)
    #         print pad_params
    #         padded_patch = pad_patch(np.squeeze(patch_img, axis=1), pad_params)
    #         for j in range(X.shape[0]):
    #             utils.imshow(X[j,0,:,:].T, Y[j,:,:].T, patch_img[j,0,:,:].T, padded_patch[j,:,:].T)
            # # import pdb 
            # # pdb.set_trace()
    # n_batches = train_iter.get_n_batches()
    # p = train_iter()
    # print n_batches
    # for i in range(n_batches):
    #     X, Y, pid = p.next()
    #     print pid
    #     print X.shape
    #     # if 'patient091' in pid:
    #     #     print X.shape, X.dtype
    #     #     print Y.shape, Y.dtype
    #     #     print pid
    #     #     for j in range(X.shape[0]):
    #     #         utils.imshow(X[j,0,:,:].T, Y[j,:,:].T)
    # print 'valid'

    # n_batches = valid_iter.get_n_batches()
    # p = valid_iter()
    # print n_batches
    # for i in range(n_batches):
    #     X, Y, pid = p.next()
    #     print pid
    #     # if 'patient091' in pid:
    #     #     print X.shape, X.dtype
    #     #     print Y.shape, Y.dtype
    #     #     print pid
    #     #     for j in range(X.shape[0]):
    #     #         utils.imshow(X[j,0,:,:].T, Y[j,:,:].T)

    # n_batches = test_iter.get_n_batches()
    # p = test_iter()
    # print n_batches
    # for i in range(n_batches):
    #     X, Y, pid = p.next()
    #     print pid
    #     # if 'patient091' in pid:
    #     #     print X.shape, X.dtype
    #     #     print Y.shape, Y.dtype
    #     #     print pid
    #     #     for j in range(X.shape[0]):
    #     #         utils.imshow(X[j,0,:,:].T, Y[j,:,:].T) 
               
    # iterator = load_data_full_test()
    # _iter = iterator.test_generator()
    # n_batches = iterator.get_n_patient_batches()
    # print n_batches
    # for i in range(n_batches):
    #     X, Y, pid, affine, hdr, roi_mask  = _iter.next()
    #     print pid
    #     batch_size = X.shape[0]
    #     n_slices = batch_size/2
    #     _,_, nrows, ncols = X.shape 

    # test = test_iter.test_generator()
    # for i in range(test_iter.get_n_patient_batches()):
    #     X, Y, pid, affine, hdr, roi_mask = test.next()
    #     print X.shape, X.dtype
    #     print Y.shape, Y.dtype
    #     print pid
    #     for j in range(X.shape[1]):
    #         utils.imshow(X[j,0,:,:].T, Y[j,:,:].T, roi_mask[j].T)

    # n_batches = test_iter.get_n_batches()
    # p = test_iter()
    # for i in range(n_batches):
    #     X, Y, pid = p.next()
    #     print X.shape, X.dtype
    #     print Y.shape, Y.dtype
    #     print pid
        # utils.imshow(X[1,0,:,:],X[1,1,:,:], Y[1,:,:],X[2,0,:,:], Y[2,:,:] )