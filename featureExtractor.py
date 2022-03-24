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
import scipy
import librosa
import soundfile

# =====================================================================================

def load_audio_file(file_path, input_fixed_length=0, params_extract=None):

    data, source_fs = soundfile.read(file=file_path)
    data = data.T

    if params_extract.get('fs') != source_fs:
        data = librosa.core.resample(data, source_fs, params_extract.get('fs'))
        print('Resampling to %d: %s' % (params_extract.get('fs'), file_path))

    if len(data) > 0:
        data = get_normalized_audio(data)
    else:
        data = np.ones((input_fixed_length, 1))
        print('File corrupted. Could not open: %s' % file_path)

    data = np.reshape(data, [-1, 1])
    return data


def modify_file_variable_length(data=None, input_fixed_length=0):

    if len(data) < input_fixed_length:
        nb_replicas = int(np.ceil(input_fixed_length / len(data)))
        data_rep = np.tile(data, (nb_replicas, 1))
        data = data_rep[:input_fixed_length]

    return data


def get_normalized_audio(y, head_room=0.005):

    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value


def get_mel_spectrogram(audio, params_extract=None):

    audio = audio.reshape([1, -1])
    window = scipy.signal.hamming(params_extract.get('win_length_samples'), sym=False)

    mel_basis = librosa.filters.mel(sr=params_extract.get('fs'),
                                    n_fft=params_extract.get('n_fft'),
                                    n_mels=params_extract.get('n_mels'),
                                    fmin=params_extract.get('fmin'),
                                    fmax=params_extract.get('fmax'),
                                    htk=False,
                                    norm=None)

    feature_matrix = np.empty((0, params_extract.get('n_mels')))
    for channel in range(0, audio.shape[0]):
        spectrogram = get_spectrogram(
            y=audio[channel, :],
            n_fft=params_extract.get('n_fft'),
            win_length_samples=params_extract.get('win_length_samples'),
            hop_length_samples=params_extract.get('hop_length_samples'),
            spectrogram_type=params_extract.get('spectrogram_type') if 'spectrogram_type' in params_extract else 'magnitude',
            center=True,
            window=window,
            params_extract=params_extract
        )

        mel_spectrogram = np.dot(mel_basis, spectrogram)
        mel_spectrogram = mel_spectrogram.T

        if params_extract.get('log'):
            mel_spectrogram = np.log10(mel_spectrogram + params_extract.get('eps'))

        feature_matrix = np.append(feature_matrix, mel_spectrogram, axis=0)

    return feature_matrix


def get_spectrogram(y,
                    n_fft=1024,
                    win_length_samples=0.04,
                    hop_length_samples=0.02,
                    window=scipy.signal.hamming(1024, sym=False),
                    center=True,
                    spectrogram_type='magnitude',
                    params_extract=None):

    if spectrogram_type == 'power':
        return np.abs(librosa.stft(y + params_extract.get('eps'),
                                   n_fft=n_fft,
                                   win_length=win_length_samples,
                                   hop_length=hop_length_samples,
                                   center=center,
                                   window=window)) ** 2
