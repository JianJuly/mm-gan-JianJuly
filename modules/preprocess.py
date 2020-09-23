"""
==========================================================
                Preprocess BRATS Data
==========================================================
AUTHOR: Anmol Sharma
AFFILIATION: Simon Fraser University
             Burnaby, BC, Canada
PROJECT: Analysis of Brain MRI Scans for Management of
         Malignant Tumors
COLLABORATORS: Anmol Sharma (SFU)
               Prof. Ghassan Hamarneh (SFU)
               Dr. Brian Toyota (VGH)
               Dr. Mostafa Fatehi (VGH)
DESCRIPTION: This file uses the previously generated data
             (using create_hdf5_file.py) and generates a
             new file  with similar structure, but after
             applying a couple of preprocessing steps.
             More specifically, the script applies the
             following operations on the data:

             1) Crop out the dark margins in the scans
                to only leave a concise brain area. For
                this a generous estimate of bounding box
                generated from the whole  dataset is used.
                For more  information, see checkLargestCropSize
                notebook.

             The code DOES NOT APPLY MEAN/VAR  normalization,
             but simply calculates the values and saves on disk.
             Check lines 140-143 for more information.

             The saved mean/var files are to be used before
             the training process.

LICENCE: Proprietary for now.
"""

import h5py
from modules.configfile import config
import numpy as np
import SimpleITK as sitk
import optparse
import logging
# from modules.mischelpers import *
from modules.dataloader import standardize
import os

logging.basicConfig(level=logging.DEBUG)

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

logger.warning('[IMPORTANT] The code DOES NOT APPLY mean/var normalization, rather it calculates it and saves to disk')
# ------------------------------------------------------------------------------------
# open existing datafile
# ------------------------------------------------------------------------------------
logger.info('opening previously generated HDF5 file.')

# open the existing datafile
hdf5_file_main = h5py.File(str(config['path_hdf5_unprocessed']), 'r')

logger.info('opened HDF5 file at {}'.format(str(config['path_hdf5_unprocessed'])))

# get the group identifier for original dataset
hdf5_file = hdf5_file_main['original_data']

# ====================================================================================

# ------------------------------------------------------------------------------------
# create new HDF5 file to hold cropped data.
# ------------------------------------------------------------------------------------
logger.info('creating new HDF5 dataset to hold cropped/normalized data')
filename = str(config['pathd_hdf5files']/'BRATS2018_Cropped.h5')
new_hdf5 = h5py.File(filename, mode='w')
logger.info('created new database at {}'.format(filename))

# create a folder group to  hold the datasets. The schema is similar to original one except for the name of the folder
# group
new_group_preprocessed = new_hdf5.create_group('preprocessed')

# create similar datasets in this file.
new_group_preprocessed.create_dataset("training_data_hgg", config['train_shape_hgg_crop'], np.float32)
new_group_preprocessed.create_dataset("training_data_hgg_pat_name", (config['train_shape_hgg_crop'][0],), dtype="S100")
new_group_preprocessed.create_dataset("training_data_segmasks_hgg", config['train_segmasks_shape_hgg_crop'], np.int16)

new_group_preprocessed.create_dataset("training_data_lgg", config['train_shape_lgg_crop'], np.float32)
new_group_preprocessed.create_dataset("training_data_lgg_pat_name", (config['train_shape_lgg_crop'][0],), dtype="S100")
new_group_preprocessed.create_dataset("training_data_segmasks_lgg", config['train_segmasks_shape_lgg_crop'], np.int16)

new_group_preprocessed.create_dataset("validation_data", config['val_shape_crop'], np.float32)
new_group_preprocessed.create_dataset("validation_data_pat_name", (config['val_shape_crop'][0],), dtype="S100")
# ====================================================================================

# just copy the patient  names directly
new_group_preprocessed['training_data_hgg_pat_name'][:] = hdf5_file['training_data_hgg_pat_name'][:]
new_group_preprocessed['training_data_lgg_pat_name'][:] = hdf5_file['training_data_lgg_pat_name'][:]
new_group_preprocessed['validation_data_pat_name'][:] = hdf5_file['validation_data_pat_name'][:]

# ------------------------------------------------------------------------------------
# start cropping process and standardization process
# ------------------------------------------------------------------------------------

# get the  file  where mean/var values are stored
# TODO: Use the config file global path, not this one.

logging.info('starting the Cropping/Normalization process.')

# only run the cropping steps on these datasets
list_crop = ['training_data_segmasks_hgg', 'training_data_hgg', 'training_data_lgg', 'training_data_segmasks_lgg', 'validation_data']

#only run the mean/var normalization on these datasets
list_std = ['training_data_hgg', 'training_data_lgg']

for cohort in list_crop:

    # we define the final shape after cropping in the config file to make it easy to access. More information available in
    # checkLargestCropSize.ipynb notebook.
    if cohort == 'training_data_hgg':
        im_np = np.empty(config['train_shape_hgg_crop'])
    elif cohort == 'training_data_lgg':
        im_np = np.empty(config['train_shape_lgg_crop'])
    elif cohort == 'validation_data':
        im_np = np.empty(config['val_shape_crop'])

    logger.info('Running on {}'.format(cohort))
    for i in range(0, hdf5_file[cohort].shape[0]):
        # cropping operation
        logger.debug('{}:- Patient {}'.format(cohort, i + 1))
        img_src = hdf5_file[cohort][i]
        coords = config['cropping_coords']
        if 'segmasks' in cohort:
            # there are no channels for segmasks
            img_dst = img_src[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]
        else:
            img_dst = img_src[:, coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]

        if cohort in list_std:
            # save the image to this numpy array
            im_np[i] = img_dst
            print(img_src.max(), img_dst.max())
        new_group_preprocessed[cohort][i] = img_dst
    # find mean and standard deviation, and apply to data. Also write the mean/std values to disk
    if cohort in list_std:
        logger.info('The dataset {} needs standardization'.format(cohort))
        _tmp, vals = standardize(im_np, findMeanVarOnly=True, saveDump=(config['pathd_hdf5files']/(cohort + '_mean_std.p')))
        logging.info('Calculated normalization values for {}:\n{}'.format(cohort, vals))
        del im_np

# ====================================================================================

hdf5_file_main.close()
new_hdf5.close()


