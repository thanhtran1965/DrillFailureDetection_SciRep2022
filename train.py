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

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import trange
import time
import pprint
import datetime
import argparse
from scipy.stats import gmean
import yaml
import tensorflow as tf

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import utils
from featureExtractor import load_audio_file, get_mel_spectrogram, modify_file_variable_length
from dataLoader import get_label_files, DataGeneratorPatch, PatchGeneratorPerFile
from model import CNN_LeakyReLU, CNN_LSTM_LeakyReLU, CNN_LSTM_Att_LeakyReLU, CNN_LSTM_Att_ReLU
from test import Evaluator

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

start = time.time()

now = datetime.datetime.now()
print("Current date and time:")
print(str(now))

# =========================================================================================================

# ==================================================================== Parser
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--params_yaml',
                    dest='params_yaml',
                    action='store',
                    required=False,
                    type=str)
args = parser.parse_args()
print('\nParameters configuration: %s\n' % str(args.params_yaml))


params = yaml.load(open(args.params_yaml))
params_dataset = params['dataset']
params_extract = params['extract']
params_learn = params['learn']
params_pred = params['predictive']

suffix_in = params['suffix'].get('in')
suffix_out = params['suffix'].get('out')

params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))
#

# ==================================================================== Dataloader
path_root_data = params_dataset.get('dataset_path')

params_path = {'path_to_features': os.path.join(path_root_data, 'features'),
               'featuredir_tr': 'audio_train_varup2/',
               'featuredir_te': 'audio_test_varup2/',
               'path_to_dataset': path_root_data,
               'audiodir_tr': 'train/',
               'audiodir_te': 'test/',
               'audio_shapedir_tr': 'audio_train_shapes/',
               'audio_shapedir_te': 'audio_test_shapes/',
               'gt_files': os.path.join(path_root_data, 'Metadata')}


params_path['featurepath_tr'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_tr'))
params_path['featurepath_te'] = os.path.join(params_path.get('path_to_features'), params_path.get('featuredir_te'))


params_path['audiopath_tr'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_tr'))
params_path['audiopath_te'] = os.path.join(params_path.get('path_to_dataset'), params_path.get('audiodir_te'))

params_path['audio_shapepath_tr'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_tr'))
params_path['audio_shapepath_te'] = os.path.join(params_path.get('path_to_dataset'),
                                                 params_path.get('audio_shapedir_te'))

params_files = {'gt_test': os.path.join(params_path.get('gt_files'), 'Drill_Dataset_Test.csv'),
                'gt_train': os.path.join(params_path.get('gt_files'), 'Drill_Dataset_Train.csv')}

train_csv = pd.read_csv(params_files.get('gt_train'))
test_csv = pd.read_csv(params_files.get('gt_test'))
filelist_audio_tr = train_csv.fname.values.tolist()
filelist_audio_te = test_csv.fname.values.tolist()

file_to_label = {params_path.get('audiopath_tr') + k: v for k, v in
                 zip(train_csv.fname.values, train_csv.label.values)}

list_labels = sorted(list(set(train_csv.label.values)))

label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}

# ==================================================================== Extractor
n_extracted_tr = 0; n_extracted_te = 0; n_failed_tr = 0; n_failed_te = 0

nb_files_tr = len(filelist_audio_tr)
if not os.path.exists(params_path.get('featurepath_tr')) or \
                len(os.listdir(params_path.get('featurepath_tr'))) < nb_files_tr*0.8:
    os.makedirs(params_path.get('featurepath_tr'))
    os.makedirs(params_path.get('featurepath_te'))

    # Training set
    for idx, f_name in enumerate(filelist_audio_tr):
        f_path = os.path.join(params_path.get('audiopath_tr'), f_name)
        if os.path.isfile(f_path) and f_name.endswith('.wav'):

            y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
            y = modify_file_variable_length(data=y,
                                            input_fixed_length=params_extract['audio_len_samples'])

            mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

            utils.save_tensor(var=mel_spectrogram,
                                out_path=os.path.join(params_path.get('featurepath_tr'),
                                                    f_name.replace('.wav', '.data')), suffix='_mel')

            utils.save_tensor(var=np.array([file_to_int[f_path]], dtype=float),
                                out_path=os.path.join(params_path.get('featurepath_tr'),
                                                    f_name.replace('.wav', '.data')), suffix='_label')

            if os.path.isfile(os.path.join(params_path.get('featurepath_tr'),
                                            f_name.replace('.wav', suffix_in + '.data'))):
                n_extracted_tr += 1
                print('%-22s: [%d/%d] of %s' % ('Extracted tr features', (idx + 1), nb_files_tr, f_path))
            else:
                n_failed_tr += 1
                print('%-22s: [%d/%d] of %s' % ('FAILING to extract tr features', (idx + 1), nb_files_tr, f_path))
        else:
            print('%-22s: [%d/%d] of %s' % ('this tr audio is in the csv but not in the folder', (idx + 1), nb_files_tr, f_path))

    print('n_extracted_tr: {0} / {1}'.format(n_extracted_tr, nb_files_tr))
    print('n_failed_tr: {0} / {1}\n'.format(n_failed_tr, nb_files_tr))

    # Testing set
    nb_files_te = len(filelist_audio_te)
    for idx, f_name in enumerate(filelist_audio_te):
        f_path = os.path.join(params_path.get('audiopath_te'), f_name)
        if os.path.isfile(f_path) and f_name.endswith('.wav'):
            y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
            y = modify_file_variable_length(data=y,
                                            input_fixed_length=params_extract['audio_len_samples'])

            mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

            utils.save_tensor(var=mel_spectrogram,
                                out_path=os.path.join(params_path.get('featurepath_te'),
                                                        f_name.replace('.wav', '.data')), suffix='_mel')

            if os.path.isfile(os.path.join(params_path.get('featurepath_te'),
                                            f_name.replace('.wav', '_mel.data'))):
                n_extracted_te += 1
                print('%-22s: [%d/%d] of %s' % ('Extracted te features', (idx + 1), nb_files_te, f_path))
            else:
                n_failed_te += 1
                print('%-22s: [%d/%d] of %s' % ('FAILING to extract te features', (idx + 1), nb_files_te, f_path))
        else:
            print('%-22s: [%d/%d] of %s' % ('this te audio is in the csv but not in the folder', (idx + 1), nb_files_te, f_path))

    print('n_extracted_te: {0} / {1}'.format(n_extracted_te, nb_files_te))
    print('n_failed_te: {0} / {1}\n'.format(n_failed_te, nb_files_te))

# ============================================================

ff_list_tr = [f for f in os.listdir(params_path.get('featurepath_tr')) if f.endswith(suffix_in + '.data') and
                  os.path.isfile(os.path.join(params_path.get('featurepath_tr'), f.replace(suffix_in, suffix_out)))]

labels_audio_train = get_label_files(filelist=ff_list_tr,
                                     dire=params_path.get('featurepath_tr'),
                                     suffix_in=suffix_in,
                                     suffix_out=suffix_out
                                     )

print('Number of clips considered as train set: {0}'.format(len(ff_list_tr)))
print('Number of labels loaded for train set: {0}'.format(len(labels_audio_train)))

tr_files, val_files = train_test_split(ff_list_tr,
                                       test_size=params_learn.get('val_split'),
                                       stratify=labels_audio_train,
                                       random_state=42
                                       )

# data generator
tr_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                    file_list=tr_files,
                                    params_learn=params_learn,
                                    params_extract=params_extract,
                                    suffix_in='_mel',
                                    suffix_out='_label',
                                    floatx=np.float32
                                    )

val_gen_patch = DataGeneratorPatch(feature_dir=params_path.get('featurepath_tr'),
                                    file_list=val_files,
                                    params_learn=params_learn,
                                    params_extract=params_extract,
                                    suffix_in='_mel',
                                    suffix_out='_label',
                                    floatx=np.float32,
                                    scaler=tr_gen_patch.scaler
                                    )

# ==================================================================== Training Model

tr_loss, val_loss = [0] * params_learn.get('n_epochs'), [0] * params_learn.get('n_epochs')
# ============================================================
model = CNN_LSTM_Att_LeakyReLU(params_learn=params_learn, params_extract=params_extract)

opt = Adam(lr=params_learn.get('lr'), beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# callbacks
early_stop = EarlyStopping(monitor='val_acc', patience=params_learn.get('patience'), min_delta=0.001, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.95, patience=5, verbose=1) # 5
checkpoint_path = 'weights/dumy_model.hdf5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

callback_list = [model_checkpoint_callback, reduce_lr, early_stop]

hist = model.fit_generator(tr_gen_patch,
                            steps_per_epoch=tr_gen_patch.nb_iterations,
                            epochs=params_learn.get('n_epochs'),
                            validation_data=val_gen_patch,
                            validation_steps=val_gen_patch.nb_iterations,
                            class_weight=None,
                            workers=4,
                            verbose=2,
                            callbacks=callback_list)



# ==================================================================== Model Prediction

print('\nCompute predictions on test set:==================================================\n')

list_preds = []
model = CNN_LSTM_Att_LeakyReLU(params_learn=params_learn, params_extract=params_extract)
model.load_weights('weights/dumy_model.hdf5')
te_files = [f for f in os.listdir(params_path.get('featurepath_te')) if f.endswith(suffix_in + '.data')]

te_preds = np.empty((len(te_files), params_learn.get('n_classes')))


te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('featurepath_te'),
                                     file_list=te_files,
                                     params_extract=params_extract,
                                     suffix_in='_mel',
                                     floatx=np.float32,
                                     scaler=tr_gen_patch.scaler
                                     )

for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):

    patches_file = te_gen_patch.get_patches_file()

    preds_patch_list = model.predict(patches_file).tolist()
    preds_patch = np.array(preds_patch_list)

    if params_pred.get('aggregate') == 'gmean':
        preds_file = gmean(preds_patch, axis=0)
    else:
        print('unkown aggregation method for prediction')
    te_preds[i, :] = preds_file


list_labels = np.array(list_labels)
pred_label_files_int = np.argmax(te_preds, axis=1)
pred_labels = [int_to_label[x] for x in pred_label_files_int]

te_files_wav = [f.replace(suffix_in + '.data', '.wav') for f in os.listdir(params_path.get('featurepath_te'))
                if f.endswith(suffix_in + '.data')]
pred = pd.DataFrame(te_files_wav, columns=["fname"])
pred['label'] = pred_labels

# ==================================================================== Model Evaluation
print('\nEvaluate ACC and print score============================================================================')

gt_test = pd.read_csv(params_files.get('gt_test'))

evaluator = Evaluator(gt_test, pred, list_labels, params_files)

print('\n=============================ACCURACY===============================================================')
print('=============================ACCURACY===============================================================\n')
evaluator.evaluate_acc()
evaluator.evaluate_acc_classwise()
evaluator.print_summary_eval()

end = time.time()
print('\n=============================Job finalized==========================================================\n')
print('Processing time: %7.2f hours' % ((end - start) / 3600.0))
print('\n====================================================================================================\n')
