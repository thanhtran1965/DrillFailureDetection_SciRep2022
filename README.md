# DSFD
# Drill Sound Failure Detection

## Software requirements:
- keras, librosa, matplotlib, numpy, pandas, scipy, sklearn, soundfile, tensorflow, tqdm. To download the dependencies: 
```
pip3 install -r requirements.txt
```

- MATLAB 2020b.

## Dataset:
Valmet AB dataset can find using this link: https://zenodo.org/record/6478574#.Ymj7ONpByUk
- Original features: [Download](https://khoavanhoceduvn-my.sharepoint.com/:u:/g/personal/2606_elibrary_su/EbcwA18Ou9FNmJHd0ACz-MoBHgCpF-U-R68wu3tOwcmW1g?e=pkwZOv)

- Augmented features: [Download](https://khoavanhoceduvn-my.sharepoint.com/:u:/g/personal/2606_elibrary_su/EWrW7FGcZqZEmQEoJh2UUDABaVQWCIzFF3tbxOdthGyMrA?e=4roj4e)

### Note: All above datasets need to download and extract to the [data/features](https://github.com/nhattruongpham/DSFD/tree/main/data/features) folder. The raw sound dataset is private, so we cannot provide the sounds according to the company's privacy policy. To support the reproduction of the experimental results, please download the above extracted/processed features that we have converted from raw sounds.

## Usage:
- First, clone the repository locally:
```
git clone https://github.com/nhattruongpham/DFSD.git
```

- Structure of repository:
```
|-data
|----features
|--------audio_test_varup2
|--------audio_train_varup2
|----Metadata
|--------Drill_Dataset_Test.csv
|--------Drill_Dataset_Train.csv
|----test
|----train
|-logs
|-weights
|-dataLoader.py
|-featureExtractor.py
|-Generate_Log-Mel_Spectrograms.ipynb
|-LICENSE
|-model.py
|-params.yaml
|-README.md
|-requirements.txt
|-ResultAnalysis.ipynb
|-Sound_Augmentation20Aug2021.m
|-test.py
|-train.py
|-utils.py
```

- Run ```Sound_Augmentation20Aug2021.m``` to generate the augmented dataset (if any).

- Run ```Generate_Log-Mel_Spectrograms.ipynb``` to extract and visualize log-Mel sepctrogram features (if any).

- Run ```python train.py -p params.yaml > logs/dumy.out``` to reproduce the several experiments and corresponding results using the proposed method. The experiments are configured by modifying lines 244 and 280 in ```train.py``` as follows:
    - CNN - Leaky ReLU:
    ```
    model = CNN_LeakyReLU(params_learn=params_learn, params_extract=params_extract)
    ```
    - CNN - LSTM - Leaky ReLU:
    ```
    model = CNN_LSTM_LeakyReLU(params_learn=params_learn, params_extract=params_extract)
    ```
    - CNN - LSTM - Att - Leaky ReLU:
    ```
    model = CNN_LSTM_Att_LeakyReLU(params_learn=params_learn, params_extract=params_extract)
    ```
    - CNN - LSTM - Att - ReLU:
    ```
    model = CNN_LSTM_Att_ReLU(params_learn=params_learn, params_extract=params_extract)
    ```

- Run ```ResultAnalysis.ipynb``` to show confusion matrix and classification report (if any).

### Note: To reproduce the experiment and result for the original dataset, please change the dataset directory! 

## Citation
If you use this code or part of it, please cite the following paper:
```
@article{tran2021detecting,
  title={Detecting Drill Failure in the Small Short-sound Drill Dataset},
  author={Tran, Thanh and Pham, Nhat Truong and Lundgren, Jan},
  journal={arXiv preprint arXiv:2108.11089},
  year={2021}
}
```
