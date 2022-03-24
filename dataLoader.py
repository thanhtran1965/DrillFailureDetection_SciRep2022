#!/usr/bin/env python

#####################################################################################
# MIT License
#
# Copyright (c) 2021 Nhat Truong Pham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# If you use this code or part of it, please cite the following paper:
# Thanh Tran, Nhat Truong Pham, and Jan Lundgren, "Detecting Drill Failure in the 
# Small Short-soundDrill Dataset", arXiv preprint arXiv:2108.11089, 2021.
#
#####################################################################################

import numpy as np
import os
import utils
from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence, to_categorical

# =====================================================================================

def get_label_files(filelist=None, dire=None, suffix_in=None, suffix_out=None):

    nb_files_total = len(filelist)
    labels = np.zeros((nb_files_total, 1), dtype=np.float32)
    for f_id in range(nb_files_total):
        labels[f_id] = utils.load_tensor(in_path=os.path.join(dire, filelist[f_id].replace(suffix_in, suffix_out)))
    return labels


class DataGeneratorPatch(Sequence):

    def __init__(self, feature_dir=None, file_list=None, params_learn=None, params_extract=None,
                 suffix_in='_mel', suffix_out='_label', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.list_fnames = file_list
        self.batch_size = params_learn.get('batch_size')
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.suffix_out = suffix_out
        self.patch_len = int(params_extract.get('patch_len'))
        self.patch_hop = int(params_extract.get('patch_hop'))

        if feature_dir is not None:
            self.get_patches_features_labels(feature_dir, file_list)

            self.features2d = self.features.reshape(-1, self.features.shape[2])

            if scaler is None:
                self.scaler = StandardScaler()
                self.features2d = self.scaler.fit_transform(self.features2d)

            else:
                self.features2d = scaler.transform(self.features2d)

            self.features = self.features2d.reshape(self.nb_inst_total, self.patch_len, self.feature_size)

        self.on_epoch_end()
        self.n_classes = params_learn.get('n_classes')

    def get_num_instances_per_file(self, f_name):

        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):

        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features_labels(self, feature_dir, file_list):

        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"
        print('Loading self.features...')

        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data') and
                          os.path.isfile(os.path.join(feature_dir, f.replace(self.suffix_in, self.suffix_out)))]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_inst_total = self.nb_inst_cum[-1]

        self.nb_iterations = int(np.floor(self.nb_inst_total / self.batch_size))

        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        self.features = np.zeros((self.nb_inst_total, self.patch_len, self.feature_size), dtype=self.floatx)
        self.labels = np.zeros((self.nb_inst_total, 1), dtype=self.floatx)

        for f_id in range(self.nb_files):
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))
        label = utils.load_tensor(in_path=os.path.join(self.feature_dir,
                                                       self.file_list[f_id].replace(self.suffix_in, self.suffix_out)))

        idx_start = self.nb_inst_cum[f_id]     
        idx_end = self.nb_inst_cum[f_id + 1]    

        idx = 0  
        start = 0 
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]

            start += self.patch_hop
            idx += 1

        self.labels[idx_start: idx_end] = label[0]

    def __len__(self):
        return self.nb_iterations

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        y_int = np.empty((self.batch_size, 1), dtype='int')
        for tt in np.arange(self.batch_size):
            y_int[tt] = int(self.labels[indexes[tt]])
        y_cat = to_categorical(y_int, num_classes=self.n_classes)

        features = self.features[indexes, np.newaxis]
        features = np.moveaxis(features, 1, -1)
        return features, y_cat

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.nb_inst_total)


class PatchGeneratorPerFile(object):

    def __init__(self, feature_dir=None, file_list=None, params_extract=None,
                 suffix_in='_mel', floatx=np.float32, scaler=None):

        self.data_dir = feature_dir
        self.floatx = floatx
        self.suffix_in = suffix_in
        self.patch_len = int(params_extract.get('patch_len')) 
        self.patch_hop = int(params_extract.get('patch_hop'))

        if feature_dir is not None:
            self.get_patches_features(feature_dir, file_list)

            self.features2d = self.features.reshape(-1, self.features.shape[2])

            self.features2d = scaler.transform(self.features2d)

            self.features = self.features2d.reshape(self.nb_patch_total, self.patch_len, self.feature_size)

    def get_num_instances_per_file(self, f_name):

        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        file_frames = float(shape[0])
        return np.maximum(1, int(np.ceil((file_frames - self.patch_len) / self.patch_hop)))

    def get_feature_size_per_file(self, f_name):

        shape = utils.get_shape(os.path.join(f_name.replace('.data', '.shape')))
        return shape[1]

    def get_patches_features(self, feature_dir, file_list):
 
        assert os.path.isdir(os.path.dirname(feature_dir)), "path to feature directory does not exist"

        self.file_list = [f for f in file_list if f.endswith(self.suffix_in + '.data')]

        self.nb_files = len(self.file_list)
        assert self.nb_files > 0, "there are no features files in the feature directory"
        self.feature_dir = feature_dir

        self.nb_inst_cum = np.cumsum(np.array(
            [0] + [self.get_num_instances_per_file(os.path.join(self.feature_dir, f_name))
                   for f_name in self.file_list], dtype=int))

        self.nb_patch_total = self.nb_inst_cum[-1]

        self.current_f_idx = 0

        self.feature_size = self.get_feature_size_per_file(f_name=os.path.join(self.feature_dir, self.file_list[0]))

        self.features = np.zeros((self.nb_patch_total, self.patch_len, self.feature_size), dtype=self.floatx)

        for f_id in range(self.nb_files):
            self.fetch_file_2_tensor(f_id)

    def fetch_file_2_tensor(self, f_id):

        mel_spec = utils.load_tensor(in_path=os.path.join(self.feature_dir, self.file_list[f_id]))

        idx_start = self.nb_inst_cum[f_id]  
        idx_end = self.nb_inst_cum[f_id + 1]  

        idx = 0  
        start = 0  
        while idx < (idx_end - idx_start):
            self.features[idx_start + idx] = mel_spec[start: start + self.patch_len]
            start += self.patch_hop
            idx += 1

    def get_patches_file(self):

        self.current_f_idx += 1

        assert self.current_f_idx <= self.nb_files, 'All the test files have been dispatched'

        features = self.features[self.nb_inst_cum[self.current_f_idx-1]: self.nb_inst_cum[self.current_f_idx], np.newaxis]
        features = np.moveaxis(features, 1, -1)

        return features

