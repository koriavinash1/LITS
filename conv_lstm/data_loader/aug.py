import numpy as np
import os
import cPickle as pickle
from collections import namedtuple
import skimage.transform
import skimage.draw
# custom
import sys
sys.path.append("../")
from data_preprocess import utils

# TODO: Check random number generator
rng = np.random.RandomState(40)
def sample_augmentation_parameters(transformation):
    # This code does random sampling from the transformation parameters
    # Random number generator
    if set(transformation.keys()) == {'patch_size', 'mm_patch_size'} or \
                    set(transformation.keys()) == {'patch_size', 'mm_patch_size', 'mask_roi'}:
        return None

    shift_x = rng.uniform(*transformation.get('translation_range_x', [0., 0.]))
    shift_y = rng.uniform(*transformation.get('translation_range_y', [0., 0.]))
    translation = (shift_x, shift_y)
    rotation = rng.uniform(*transformation.get('rotation_range', [0., 0.]))
    shear = rng.uniform(*transformation.get('shear_range', [0., 0.]))
    roi_scale = rng.uniform(*transformation.get('roi_scale_range', [1., 1.]))
    z = rng.uniform(*transformation.get('zoom_range', [1., 1.]))
    zoom = (z, z)

    if 'do_flip' in transformation:
        if type(transformation['do_flip']) == tuple:
            flip_x = rng.randint(2) > 0 if transformation['do_flip'][0] else False
            flip_y = rng.randint(2) > 0 if transformation['do_flip'][1] else False
        else:
            flip_x = rng.randint(2) > 0 if transformation['do_flip'] else False
            flip_y = False
    else:
        flip_x, flip_y = False, False

    sequence_shift = rng.randint(30) if transformation.get('sequence_shift', False) else 0

    return namedtuple('Params', ['translation', 'rotation', 'shear', 'zoom',
                                 'roi_scale',
                                 'flip_x', 'flip_y',
                                 'sequence_shift'])(translation, rotation, shear, zoom,
                                                    roi_scale,
                                                    flip_x, flip_y,
                                                    sequence_shift)

def fast_warp(img, tf, output_shape, mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params  # tf._matrix is
    # TODO: check if required
    # mode='symmetric'
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)


def build_rescale_transform(downscale_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=(downscale_factor, downscale_factor))
    # centering
    shift_x = cols / (2.0 * downscale_factor) - tcols / 2.0
    shift_y = rows / (2.0 * downscale_factor) - trows / 2.0
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds

def make_roi_mask(img_shape, roi_center, roi_radii):
    """
    Makes 2D ROI mask for one slice
    :param data:
    :param roi:
    :return:
    """
    mask = np.zeros(img_shape)
    mask[max(0, roi_center[0] - roi_radii[0]):min(roi_center[0] + roi_radii[0], img_shape[0]),
    max(0, roi_center[1] - roi_radii[1]):min(roi_center[1] + roi_radii[1], img_shape[1])] = 1
    return mask

def make_circular_roi_mask(img_shape, roi_center, roi_radii):
    mask = np.zeros(img_shape)
    rr, cc = skimage.draw.ellipse(roi_center[0], roi_center[1], roi_radii[0], roi_radii[1], img_shape)
    mask[rr, cc] = 1.
    return mask


def make_elliptical_roi_mask(img_shape, roi_center, roi_radii):
    full_mask = np.empty(img_shape[:2]+(0,))
    for i in range(img_shape[2]):
        factor = 2*i
        mask = np.zeros(img_shape[:2])
        rr, cc = skimage.draw.ellipse(roi_center[0], roi_center[1],
        (roi_radii[0]-factor), (roi_radii[1]-factor), img_shape[:2])
        mask[rr, cc] = 1.
        mask = np.expand_dims(mask, axis=2)
        full_mask = np.append(full_mask, mask, axis=2)
    return full_mask

def build_augmentation_transform(rotation=0, shear=0, translation=(0, 0), flip_x=False, flip_y=False, zoom=(1.0, 1.0)):
    if flip_x:
        shear += 180  # shear by 180 degrees is equivalent to flip along the X-axis
    if flip_y:
        shear += 180
        rotation += 180

    tform_augment = skimage.transform.AffineTransform(scale=(1. / zoom[0], 1. / zoom[1]), rotation=np.deg2rad(rotation),
                                                      shear=np.deg2rad(shear), translation=translation)
    return tform_augment


def build_shift_center_transform(image_shape, center_location, patch_size):
    """Shifts the center of the image to a given location.
    This function tries to include as much as possible of the image in the patch
    centered around the new center. If the patch around the ideal center
    location doesn't fit within the image, we shift the center to the right so
    that it does.
    params in (i,j) coordinates !!!
    """
    if center_location[0] < 1. and center_location[1] < 1.:
        center_absolute_location = [
            center_location[0] * image_shape[0], center_location[1] * image_shape[1]]
    else:
        center_absolute_location = [center_location[0], center_location[1]]

    # Check for overlap at the edges
    center_absolute_location[0] = max(
        center_absolute_location[0], patch_size[0] / 2.0)
    center_absolute_location[1] = max(
        center_absolute_location[1], patch_size[1] / 2.0)

    center_absolute_location[0] = min(
        center_absolute_location[0], image_shape[0] - patch_size[0] / 2.0)

    center_absolute_location[1] = min(
        center_absolute_location[1], image_shape[1] - patch_size[1] / 2.0)

    # Check for overlap at both edges
    if patch_size[0] > image_shape[0]:
        center_absolute_location[0] = image_shape[0] / 2.0
    if patch_size[1] > image_shape[1]:
        center_absolute_location[1] = image_shape[1] / 2.0

    # Build transform
    new_center = np.array(center_absolute_location)
    translation_center = new_center - 0.5
    translation_uncenter = -np.array((patch_size[0] / 2.0, patch_size[1] / 2.0)) - 0.5
    return (
        skimage.transform.SimilarityTransform(translation=translation_center[::-1]),
        skimage.transform.SimilarityTransform(translation=translation_uncenter[::-1]))

def normalize_contrast_zmuv_4D(data, z=2):
    """
    Normalize contrast across volume+time
    """
    mean = np.mean(data)
    std = np.std(data)
    for i in xrange(data.shape[2]):
        for j in xrange(data.shape[3]):
            img = data[:,:,i,j]
            img = ((img - mean) / (2 * std * z) + 0.5)
            data[:,:,i,j] = np.clip(img, -0.0, 1.0)

def normalize_contrast_zmuv_3D(data, z=2):
    """
    Normalize contrast across volume
    """
    mean = np.mean(data)
    std = np.std(data)
    for i in xrange(data.shape[2]):
        img = data[:, :, i]
        img = ((img - mean) / (2 * std * z) + 0.5)
        data[:, :, i] = np.clip(img, -0.0, 1.0)

def normalize_contrast_zmuv_2D(data, z=2):
    """
    Normalize contrast across volume
    """
    mean = np.mean(data)
    std = np.std(data)
    # img = ((data - mean) / (2 * std * z) + 0.5)
    # img = np.clip(img, -0.0, 1.0)
    # TODO: check no clipping
    img = (data - mean) / (std)
    return img

def normalize_contrast_2D(data, z=2):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    img = (data - _min) / (_max-_min)
    return img
#TODO: Normalization schemes:

def multilabel_binarize(image_2D, nlabel):
    """
    Binarize multilabel images and return stack of binary images
    Returns: Tensor of shape: Bin_Channels* Image_shape(3D tensor)
    TODO: Labels are assumed to discreet in steps from -> 0,1,2,...,nlabel-1
    """
    labels = range(nlabel)
    out_shape = (len(labels),) + image_2D.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_2D == label, bin_img_stack[label], 0)
    return bin_img_stack

def multilabel_transform(img, tf, output_shape, nlabel, mode='constant', order=0):
    """
    Binarize images do apply transform on each of the binary images and take argmax while
    doing merge operation
    Order -> 0 : nearest neighbour interpolation
    """
    bin_img_stack = multilabel_binarize(img, nlabel)
    n_labels = len(bin_img_stack)
    tf_bin_img_stack = np.zeros((n_labels,) + output_shape, dtype='uint8')
    for label in xrange(n_labels):
        tf_bin_img_stack[label] = fast_warp(bin_img_stack[label], tf, output_shape=output_shape, mode=mode, order=order)
    # Do merge operation along the axis = 0
    return np.argmax(tf_bin_img_stack, axis=0)


def roi_patch_transform_norm(data, transformation, nlabel, random_augmentation_params=None,
                             mm_center_location=(.5, .4), mm_patch_size=(128, 128), mask_roi=False, uniform_scale=False):

    # Input data dimension is of shape: X * Y * Z * t
    patch_size = transformation['patch_size']
    mm_patch_size = transformation.get('mm_patch_size', mm_patch_size)
    mask_roi = transformation.get('mask_roi', mask_roi)
    # Output data shape:
    nframe_ED = int(data['ED'])
    nframe_ES = int(data['ES'])
    nslices = data['4D'].shape[2]
    ed_img = data['4D'][:, :, :, nframe_ED]
    ed_gt = data['ED_GT']
    es_img = data['4D'][:, :, :, nframe_ES]
    es_gt = data['ES_GT']
    out_shape = patch_size + (nslices,)
    ed_img_out = np.zeros(out_shape, dtype='float32')
    es_img_out = np.zeros(out_shape, dtype='float32')
    ed_gt_out = np.zeros(out_shape, dtype='uint8')
    es_gt_out = np.zeros(out_shape, dtype='uint8')
    # FFT Harmonics
    # fft_0 = utils.read_fft_volume(data['4D'], harmonic=0)
    # fft_1 = utils.read_fft_volume(data['4D'], harmonic=1)
    # fft0_img_out = np.zeros(out_shape, dtype='float32')
    # fft1_img_out = np.zeros(out_shape, dtype='float32')

    # pixel spacing in X and Y
    pixel_spacing = data['pixel_spacing'][:2]
    roi_center = data['roi_center']
    roi_radii = data['roi_radii']

    # if random_augmentation_params=None -> sample new params
    # if the transformation implies no augmentations then random_augmentation_params remains None
    if not random_augmentation_params:
        random_augmentation_params = sample_augmentation_parameters(transformation)
        # print random_augmentation_params
    # build scaling transformation
    assert pixel_spacing[0] == pixel_spacing[1]
    current_shape = data['4D'].shape[:2]

    # scale ROI radii and find ROI center in normalized patch
    if roi_center:
        mm_center_location = tuple(int(r * ps) for r, ps in zip(roi_center, pixel_spacing))

    # scale the images such that they all have the same scale
    norm_rescaling = 1./ pixel_spacing[0] if uniform_scale else 1
    mm_shape = tuple(int(float(d) * ps) for d, ps in zip(current_shape, pixel_spacing))

    tform_normscale = build_rescale_transform(downscale_factor=norm_rescaling,
                                              image_shape=current_shape, target_shape=mm_shape)

    tform_shift_center, tform_shift_uncenter = build_shift_center_transform(image_shape=mm_shape,
                                                                            center_location=mm_center_location,
                                                                            patch_size=mm_patch_size)
    patch_scale = max(1. * mm_patch_size[0] / patch_size[0],
                      1. * mm_patch_size[1] / patch_size[1])
    tform_patch_scale = build_rescale_transform(patch_scale, mm_patch_size, target_shape=patch_size)

    total_tform = tform_patch_scale + tform_shift_uncenter + tform_shift_center + tform_normscale

    # build random augmentation
    if random_augmentation_params:
        augment_tform = build_augmentation_transform(rotation=random_augmentation_params.rotation,
                                                     shear=random_augmentation_params.shear,
                                                     translation=random_augmentation_params.translation,
                                                     flip_x=random_augmentation_params.flip_x,
                                                     flip_y=random_augmentation_params.flip_y,
                                                     zoom=random_augmentation_params.zoom)
        total_tform = tform_patch_scale + tform_shift_uncenter + augment_tform + tform_shift_center + tform_normscale
        # print total_tform.params
    # apply transformation per image
    # TODO: check Normalization schemes
    for slice_n in xrange(nslices):
        # normalize_contrast_2D
        ed_img_out[:, :, slice_n] = fast_warp(normalize_contrast_zmuv_2D(ed_img[:, :, slice_n]), total_tform, output_shape=patch_size, mode='symmetric')
        es_img_out[:, :, slice_n] = fast_warp(normalize_contrast_zmuv_2D(es_img[:, :, slice_n]), total_tform, output_shape=patch_size, mode='symmetric')
        # fft0_img_out[:, :, slice_n] = fast_warp(fft_0[:, :, slice_n], total_tform, output_shape=patch_size)
        # fft1_img_out[:, :, slice_n] = fast_warp(fft_1[:, :, slice_n], total_tform, output_shape=patch_size)
        ed_gt_out[:, :, slice_n] = multilabel_transform(ed_gt[:, :, slice_n], total_tform, patch_size, nlabel)
        es_gt_out[:, :, slice_n] = multilabel_transform(es_gt[:, :, slice_n], total_tform, patch_size, nlabel)
    # normalize_contrast_zmuv_2D(ed_img_out)
    # normalize_contrast_zmuv_2D(es_img_out)

    # apply transformation to ROI and mask the images
    if roi_center and roi_radii and mask_roi:
        roi_scale = random_augmentation_params.roi_scale if random_augmentation_params else 1.6  # augmentation
        roi_zoom = random_augmentation_params.zoom if random_augmentation_params else pixel_spacing
        rescaled_roi_radii = (roi_scale * roi_radii[0], roi_scale * roi_radii[1])
        out_roi_radii = (int(roi_zoom[0] * rescaled_roi_radii[0] * pixel_spacing[0] / patch_scale),
                         int(roi_zoom[1] * rescaled_roi_radii[1] * pixel_spacing[1] / patch_scale))
        roi_mask = make_circular_roi_mask(out_shape, (patch_size[0] / 2, patch_size[1] / 2), out_roi_radii)
        ed_img_out *= roi_mask
        es_img_out *= roi_mask
        # fft0_img_out *= roi_mask
        # fft1_img_out *= roi_mask

    if random_augmentation_params:
        targets_zoom_factor = random_augmentation_params.zoom[0] * random_augmentation_params.zoom[1]
    else:
        targets_zoom_factor = pixel_spacing[0]*pixel_spacing[1]

    output = {}
    output['ED'] = ed_img_out
    output['ES'] = es_img_out
    # output['FFT_0'] = fft0_img_out
    # output['FFT_1'] = fft1_img_out
    output['ED_GT'] = ed_gt_out
    output['ES_GT'] = es_gt_out
    output['zoom'] = targets_zoom_factor
    return output

def roi_patch_norm(data, mask_roi=False):

    # Input data dimension is of shape: X * Y * Z * t
    # Output data shape:
    nframe_ED = int(data['ED'])
    nframe_ES = int(data['ES'])
    nslices = data['4D'].shape[2]
    ed_img = data['4D'][:, :, :, nframe_ED]
    es_img = data['4D'][:, :, :, nframe_ES]
    ed_gt = data['ED_GT']
    es_gt = data['ES_GT']
    ed_img_out = np.zeros_like(ed_img, dtype='float32')
    es_img_out = np.zeros_like(es_img, dtype='float32')

    # pixel spacing in X and Y
    pixel_spacing = data['pixel_spacing'][:2]
    roi_center = data['roi_center']
    roi_radii = data['roi_radii']
    # print roi_center
    # build scaling transformation
    assert pixel_spacing[0] == pixel_spacing[1]

    # apply transformation per image
    # TODO: check Normalization schemes
    roi_mask = np.empty((ed_gt.shape[:2]+(0,)), dtype=np.uint8)
    for slice_n in xrange(nslices):
        ed_img_out[:, :, slice_n] = normalize_contrast_zmuv_2D(ed_img[:, :, slice_n])
        es_img_out[:, :, slice_n] = normalize_contrast_zmuv_2D(es_img[:, :, slice_n])       
    # apply transformation to ROI
    if roi_center and roi_radii:
        roi_scale = 1  # augmentation
        # roi_zoom = pixel_spacing
        roi_zoom = (2.3, 2)
        rescaled_roi_radii = (roi_scale * roi_radii[0], roi_scale * roi_radii[1])
        out_roi_radii = (int(roi_zoom[0] * rescaled_roi_radii[0]),
                         int(roi_zoom[1] * rescaled_roi_radii[1]))
        roi_mask = make_elliptical_roi_mask(ed_img.shape, roi_center, out_roi_radii)
        # Mask the images
        if mask_roi:
            ed_img_out *= roi_mask
            es_img_out *= roi_mask
            fft_0 *= roi_mask
            fft_1 *= roi_mask

    #     print roi_mask.shape
    # utils.imshow(roi_mask[:,:,0])
    targets_zoom_factor = pixel_spacing[0]*pixel_spacing[1]
    output = {}
    # FFT
    # fft_0 = utils.read_fft_volume(data['4D'], harmonic=0)
    # fft_1 = utils.read_fft_volume(data['4D'], harmonic=1)
    # # TODO: Optimize on post-processing of ROI mask and multiplication       
    # # output['FFT_0'] = fft_0
    # output['FFT_1'] = fft_1
    output['ROI'] = roi_mask     
    output['ED'] = ed_img_out
    output['ES'] = es_img_out
    output['ED_GT'] = ed_gt
    output['ES_GT'] = es_gt
    # output['FFT_1_ROI'] = np.expand_dims(contour_image(fft_1), axis=2)*roi_mask
    output['zoom'] = targets_zoom_factor
    return output



if __name__ == "__main__":
    # Pickle file path
    fulldata_pickled_path = '/home/bmi/Documents/mak/Cardiac_dataset/ACDC/dataset/processed_data/pickled/complete_patient_data.pkl'
    train_pickled_path = '../../dataset/processed_data/pickled/train_patient_data.pkl'
    validation_pickled_path = '../../dataset/processed_data/pickled/validation_patient_data.pkl'
    test_pickled_path = '../../dataset/processed_data/pickled/test_patient_data.pkl'
    test_patient_data = utils.load_pkl(fulldata_pickled_path)

    # Set patch extraction parameters
    size1 = (256, 256)
    size2 = (128, 128)
    size3 = (180, 180)
    patch_size = size1
    mm_patch_size = size1
    train_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
        # 'rotation_range': (-180, 180),
        # 'translation_range_x': (-15, 15),
        # 'translation_range_y': (-15, 15),
        # 'zoom_range': (.5, 2.0),
        # 'do_flip': (True, True),
    }
    test_transformation_params = {
        'patch_size': patch_size,
        'mm_patch_size': mm_patch_size,
    }
    # # for Test
    # for patient_id in test_patient_data.keys():
    #     patient_data = test_patient_data[patient_id]
    #     out_data = roi_patch_norm(patient_data, mask_roi=False)
    #     print patient_id
    #     print out_data['ED'].shape
    #     print out_data['ED_GT'].shape
    #     print out_data['ROI'].shape
    #     import matplotlib
    #     matplotlib.use("TkAgg")
    #     import matplotlib.pyplot as plt
    #     for i in range(out_data['ED'].shape[2]):
    #         utils.imshow(out_data['ED'][:,:,i].T, out_data['ED_GT'][:,:,i].T, out_data['ROI'][:,:,i].T,
    #             out_data['ED_GT'][:,:,i].T*(1-out_data['ROI'][:,:,i].T))
    # For train generator
    for patient_id in test_patient_data.keys():
        if patient_id == 'patient021':
            patient_data = test_patient_data[patient_id]
            nlabel = 4
            out_data = roi_patch_transform_norm(patient_data, train_transformation_params, nlabel, uniform_scale=False)
            print patient_id
            print out_data['ED'].shape
            print out_data['ED_GT'].shape
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            for i in range(out_data['ED'].shape[2]):
                utils.imshow(out_data['ED'][:,:,i].T, out_data['ED_GT'][:,:,i].T)

            # for i in range(out_data['ED'].shape[2]):
            #     pass
                # utils.imshow(edg[:,:,i].T, bin_morph[:,:,i].T, fill[:,:].T)
                # utils.imshow(patient_data['4D'][:,:,i,0].T, patient_data['ED_GT'][:,:,i].T, out_data['ED'][:,:,i].T, out_data['ED_GT'][:,:,i].T, patient_data['ES_GT'][:,:,i].T, out_data['ES'][:,:,i].T, out_data['ES_GT'][:,:,i].T)
                # utils.imshow(patient_data['4D'][:,:,i,0].T, patient_data['ED_GT'][:,:,i].T, out_data['ED'][:,:,i].T, out_data['ED_GT'][:,:,i].T)
                # out_data1=roi_patch_transform_inv(patient_data, out_data['ED_GT'], 4)
                # utils.imshow(patient_data['4D'][:,:,i,0].T, out_data['ED'][:,:,i].T, out_data1['ED'][:,:,i].T)

                # utils.imshow(out_data['ED'][:,:,i].T, out_data['FFT_1'][:,:,i].T, out_data['ED_GT'][:,:,i].T, out_data['ROI'][:,:,i].T,
                #     out_data['ED_GT'][:,:,i].T * out_data['ROI'][:,:,i].T)
                # import matplotlib.pyplot as plt
                # roi_center = patient_data['roi_center']
                # roi_radii = patient_data['roi_radii']
                # plt.imshow(out_data['ED_GT'][:,:,i].T)
                # plt.plot([roi_center[0]], [roi_center[1]], marker='o', markersize=3, color="red")
                # circle1 = plt.Circle(roi_center, roi_radii[0], color='r', fill=False)
                # circle2 =plt.Circle(roi_center, roi_radii[1], color='r', fill=False)
                # ax = plt.gca()
                # ax.add_artist(circle1)
                # ax.add_artist(circle2)
                # plt.savefig('test{}'.format(i))
                # plt.close()
