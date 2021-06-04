import os
import time
import h5py
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
import itertools
from math import floor
from operator import truediv


def tensorify_melgram(melgram):
    tensor = melgram[np.newaxis, :, :, np.newaxis]
    return tensor


#number of genre labels
num_genres = 10
genres_list = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
#number of songs desired (per genre), max 100
num_songs_train = 80
num_songs_test = 100 - num_songs_train
channel = 1


def generate_user_input(filepath):
    filepath = filepath
    filepath = filepath + song + '.wav'
    y, sr = librosa.load(filepath)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    (height, width) = mel_spect.shape
    if (height == 128) and (width == 647):
        pass
    else:
        if width > 647:
            mel_spect = mel_spect[:,0:647]
        elif width < 647:
            missing = 647 - width
            new_cols = np.zeros((128,missing))
            mel_spect = np.concatenate((mel_spect, new_cols), axis=1)
    counter = 1
    tensor = tensorify_melgram(mel_spect)
    #print("tensor dimensions: "+str(tensor.shape))
    mel_array = np.concatenate((mel_array, tensor), axis=0)
    return (mel_array, counter)


def generate_dataset_train():
    #generates an array of n mel spectrograms and a list of n labels, for each
    #mel spectrogram respectively

    mel_array = np.zeros((0,128,647,1))
    labels = []
    song_counter = 0
    genre_counter = 0
    counter = 0
    missed = 0
    while genre_counter < num_genres and song_counter < num_songs_train:
        genre = genres_list[genre_counter]
        filepath = './genres/' + genre + '/' + genre + '.000'
        if song_counter < 10:
            song = '0' + str(song_counter)
        else:
            song = str(song_counter)
        filepath = filepath + song + '.wav'
        y, sr = librosa.load(filepath)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        (height, width) = mel_spect.shape
        if (height == 128) and (width == 647):
            pass
        else:
            if width > 647:
                mel_spect = mel_spect[:,0:647]
            elif width < 647:
                missing = 647 - width
                new_cols = np.zeros((128,missing))
                mel_spect = np.concatenate((mel_spect, new_cols), axis=1)
        counter = counter + 1
        tensor = tensorify_melgram(mel_spect)
        mel_array = np.concatenate((mel_array, tensor), axis=0)
        labels.append(genre_counter)
        if song_counter < (num_songs_train - 1):
            song_counter = song_counter + 1
        else:
            genre_counter = genre_counter + 1
            song_counter = 0
    print("mel_array size: "+str(mel_array.shape))
    return (mel_array, labels, counter)


def generate_dataset_test():
    #generates an array of n mel spectrograms and a list of n labels, for each
    #mel spectrogram respectively
    mel_array = np.zeros((0,128,647,1), dtype=np.float32)
    labels = []
    song_counter = 99
    genre_counter = 0
    counter = 0
    while genre_counter < num_genres and song_counter >= num_songs_test:
        genre = genres_list[genre_counter]
        filepath = './genres/' + genre + '/' + genre + '.000'
        if song_counter < 10:
            song = '0' + str(song_counter)
        else:
            song = str(song_counter)
        filepath = filepath + song + '.wav'
        y, sr = librosa.load(filepath)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=2048, hop_length=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        (height, width) = mel_spect.shape
        if (height == 128) and (width == 647):
            pass
        else:
            if width > 647:
                mel_spect = mel_spect[:,0:647]
            elif width < 647:
                missing = 647 - width
                new_cols = np.zeros((128,missing))
                mel_spect = np.concatenate((mel_spect, new_cols), axis=1)
        tensor = tensorify_melgram(mel_spect)
        mel_array = np.concatenate((mel_array, tensor), axis=0)
        labels.append(genre_counter)
        counter = counter + 1
        if song_counter > num_songs_train:
            song_counter = song_counter - 1
        else:
            genre_counter = genre_counter + 1
            song_counter = 99
    print("test mel_array size: "+str(mel_array.shape))
    return (mel_array, labels, counter)


def get_label(prediction_array, toggle):
    if toggle == 0:
        max = 0
        idx = 0
        for i in range(len(prediction_array)):
            score = prediction_array[i]
            if score > max:
                max = score
                idx = i
            else:
                pass
        return idx
    elif toggle == 1:
        #toggle left in to be able to use different methods of label extraction in the future
        pass
    

def save_data(data, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)
        

def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('labels'))
        num_frames = np.array(hf.get('num_frames'))
    return data, labels, num_frames


def save_dataset(path, data, labels, num_frames):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('num_frames', data=num_frames)

        
def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print(name, ':', '%5.3f  ' % score, '   ')


def predict_label(preds):
    labels=preds.argsort()[::-1]
    return labels[0]


def load_gt(path):
    with open(path, "r") as insTest:
        gt_total = []
        for lineTest in insTest:
            gt_total.append(int(lineTest))
        gt_total = np.array(gt_total)
        # print test_numFrames_total
    return gt_total


def plot_confusion_matrix(cnf_matrix, classes, title):

    cnfm_suma=cnf_matrix.sum(1)
    cnfm_suma_matrix = np.repeat(cnfm_suma[:,None],cnf_matrix.shape[1],axis=1)

    cnf_matrix=10000*cnf_matrix/cnfm_suma_matrix
    cnf_matrix=cnf_matrix/(100*1.0)
    print(cnf_matrix)

    #print map(truediv,cnf_matrix, cnfm_suma_matrix)

    fig=plt.figure()
    cmap=plt.cm.Blues
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    #plt.show()
    fig.savefig(title)
