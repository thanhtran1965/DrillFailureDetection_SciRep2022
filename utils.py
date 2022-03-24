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
import os, re
import matplotlib
matplotlib.use('agg')

# =====================================================================================

def save_tensor(var, out_path=None, suffix='_mel'):

    assert os.path.isdir(os.path.dirname(out_path)), "path to save tensor does not exist"
    var.tofile(out_path.replace('.data', suffix + '.data'))
    save_shape(out_path.replace('.data', suffix + '.shape'), var.shape)


def load_tensor(in_path, suffix=''):

    assert os.path.isdir(os.path.dirname(in_path)), "path to load tensor does not exist"
    f_in = np.fromfile(in_path.replace('.data', suffix + '.data'))
    shape = get_shape(in_path.replace('.data', suffix + '.shape'))
    f_in = f_in.reshape(shape)
    return f_in


def save_shape(shape_file, shape):

    with open(shape_file, 'w') as fout:
        fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')


def get_shape(shape_file):

    with open(shape_file, 'rb') as f:
        line=f.readline().decode('ascii')
        if line.startswith('#'):
            shape=tuple(map(int, re.findall(r'(\d+)', line)))
            return shape
        else:
            raise IOError('Failed to find shape in file')


def get_num_instances_per_file(f_name, patch_len=25, patch_hop=12):

    shape = get_shape(os.path.join(f_name.replace('.data', '.shape')))
    file_frames = float(shape[0])
    return np.maximum(1, int(np.ceil((file_frames-patch_len)/patch_hop)))


def get_feature_size_per_file(f_name):

    shape = get_shape(os.path.join(f_name.replace('.data', '.shape')))
    return shape[1]


def make_sure_isdir(pre_path, _out_file):

    full_path = os.path.join(pre_path, _out_file)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    return full_path
