%% Set datafolder to the location of the data.
datafolder = 'E:\BackupData\2stYear2020\4.Papers\1.IEEE-ACMTransaction_InAudioSpeechAndSignalProcessing\Revision_June2021\Drill_Dataset_ValmetAB_11June2021\Other\';
% Create an audio datastore to point to the audio data
ADS        = audioDatastore(datafolder, "Includesubfolders",true,'FileExtensions','.wav');  

%% Show the number of sounds in the dataset
numFilesInDataset = numel(ADS.Files)

%% Create an audioDataAugmenter for volume control, noise corruption and time shifting
% 'independent' –– Augmentation algorithms are applied independently (in parallel).
% Source of augmentation parameters, specified as 'random': Augmentation algorithms are applied probabilistically using a probability parameter and a range parameter.
aug = audioDataAugmenter( ...
    "AugmentationMode","independent", ...
    "AugmentationParameterSource","specify", ...
    "ApplyTimeStretch",false, ...
    "ApplyPitchShift",false, ...
    "ApplyAddNoise",false, ...
    "SNR",0, ...
    "VolumeGain",2, ...
    "TimeShift",0.005)

%% To augment the audio dataset, create two augmentations of each file and then write the augmentations as WAV files.
while hasdata(ADS)
    [audioIn,info] = read(ADS);
    
    data = augment(aug,audioIn,info.SampleRate);
    
    [~,fn] = fileparts(info.FileName);
    for i = 1:size(data,1)
        augmentedAudio = data.Audio{i};
        
        % If augmentation caused an audio signal to have values outside of -1 and 1, 
        % normalize the audio signal to avoid clipping when writing.
        if max(abs(augmentedAudio),[],'all')>1
            augmentedAudio = augmentedAudio/max(abs(augmentedAudio),[],'all');
        end
        
        audiowrite(sprintf('%s_aug%d.wav',fn,i),augmentedAudio,info.SampleRate)
    end
end

%% Confirm that the number of files in the dataset is double the original number of files.
augmentedADS = audioDatastore(pwd)
numFilesInAugmentedDataset = numel(augmentedADS.Files)

%% Load data to vizualize
% Link to the original audio
[audioIn,fs] = audioread("E:\BackupData\2stYear2020\4.Papers\1.IEEE-ACMTransaction_InAudioSpeechAndSignalProcessing\Revision_June2021\AugmentedDataset18Aug\Normal\normal_56.wav");
sound(audioIn,fs)

%% Load the augmented audio and visualize in the same figure
[audioAug1,fs] = audioread("E:\BackupData\2stYear2020\4.Papers\1.IEEE-ACMTransaction_InAudioSpeechAndSignalProcessing\Revision_June2021\AugmentedDataset18Aug\Normal\normal_56.wav");
sound(audioAug1,fs)

t = (0:(numel(audioIn)-1))/fs;
taug = (0:(numel(audioAug1)-1))/fs;
plot(t,audioIn,taug,audioAug1)
legend("Original sound","Volume control")
ylabel("Amplitude")
xlabel("Time (s)")