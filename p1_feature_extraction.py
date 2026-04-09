import os  # needed navigate the system to get the input data
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from radiomics.featureextractor import RadiomicsFeatureExtractor
from pandas import DataFrame
import math
import cv2
import threading
import requests
import time
import multiprocessing


def feature_extractor_nii(nimg,mak): # nimg: the directory of dcm files,mak: mha or nrrd path
    mask_name = mak.split('/')[-1].split('.')[0]
    print(mask_name)
    img = sitk.ReadImage(nimg)

    mask = sitk.ReadImage(mak)
    mask_array = sitk.GetArrayFromImage(mask)
    mask.SetOrigin(img.GetOrigin())
    mask.SetSpacing(img.GetSpacing())

    mask.SetDirection(img.GetDirection())
    settings = {'binWidth': 25,
                'interpolator': sitk.sitkBSpline,
                'label': int(np.unique(mask_array)[-1]),
                'resampledPixelSpacing': (1,1,1),
                'geometryTolerance': 0.0001,
              }
    extractor = RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableAllFeatures()
    features = extractor.execute(img, mask)

    feature_data = np.hstack((nimg, mak, list(features.values())))
    all_feature_column = np.hstack(('img_path', 'mak_path', [p for p in list(features.keys())]))
    all_feature = DataFrame(data=feature_data, index=all_feature_column).T
    all_feature.to_csv(mask_name + '_' + 'features.csv')

    return all_feature



root_path = './data/nifti_roi'

data_empty = []
patient_all = np.sort(os.listdir(root_path))
patient_idx = [i.split('_')[0] for i in patient_all]
patient_list = list(set(patient_idx))
patient_list.sort()
series_all = ['DWI', 'T2', 'T1C']

slice_length = int(len(patient_list)/4)
def feature_ex(num):

    patient_list_idx = patient_list[slice_length*(num-1):slice_length*(num)]
    for patient in patient_list_idx:
        threads = []
        for series in series_all:
        
            image_i = patient + '_' + series + '.nii.gz'
            image_path = os.path.join(root_path, image_i)

            if os.path.exists(image_path):
                roi_i = patient + '_' + series + '_ROI.nii.gz'
                roi_path = os.path.join(root_path, roi_i)
                print(roi_path)
                if os.path.exists(roi_path):
                    # features = feature_extractor_nii(image_path, roi_path)

                    t = threading.Thread(target=feature_extractor_nii, args=(image_path, roi_path))
                    t.start()
                    threads.append(t)

        for thread in threads:
             thread.join()

processes = []
lop_size = [1,2,3,4]
# lop_size = [1, 2]
p = multiprocessing.Pool()
p.map(feature_ex,lop_size)
p.close()
p.join()

