from keras import backend as K
import os
import time
import h5py
import sys
#from tagger_net import MusicTaggerCRNN
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
from math import floor
from cnn import MusicGenre_CNN
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, load_gt, plot_confusion_matrix, generate_dataset_train, generate_dataset_test, get_label

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Toggles
TRAIN = 1
TEST = 0

SAVE_MODEL = 0
SAVE_WEIGHTS = 0

LOAD_MODEL = 0
LOAD_WEIGHTS = 0

# Dataset
MULTIFRAMES = 0
SAVE_DB = 0
LOAD_DB = 0

# Model parameters
n_classes = 10
n_epochs = 20
batch_size = 80

time_elapsed = 0

# GTZAN Dataset Tags
tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
tags = np.array(tags)

# Paths
model_name = "cnn"
model_path = "trained_" + model_name + "/"
weights_path = model_path + "weights/"

print("Generating training dataset...")
x_train, y_train, n_train = generate_dataset_train()
print("Generating testing dataset...")
x_test, y_test, n_test = generate_dataset_test()

y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

# Initialize model
model = MusicGenre_CNN(weights=None, input_tensor=(128, 647, 1))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

if LOAD_WEIGHTS:
    model.load_weights(weights_path+model_name+'_epoch_40.h5')

# Print summary of the model
model.summary()

# Save model architecture
'''
if SAVE_MODEL:
    json_string = model.to_json()
    f = open(model_path+model_name+".json", 'w')
    f.write(json_string)
    f.close()
'''

# Train model
if TRAIN:
    #try:
    print ("Training the model")
    f_train = open(model_path+"scores_train.txt", 'w')
    f_test = open(model_path+"scores_test.txt", 'w')
    f_scores = open(model_path+"scores.txt", 'w')
    for epoch in range(1,n_epochs+1):
        t0 = time.time()
        # y_train = np_utils.to_categorical(y_train)
        # y_test = np_utils.to_categorical(y_test)
        print ("Number of epoch: " +str(epoch)+"/"+str(n_epochs))
        sys.stdout.flush()
        #print("shape of x_train: "+str(x_train.shape)+", shape of y_train: "+str(y_train.shape))
        #print("shape of x_test: "+str(x_test.shape)+", shape of y_test: "+str(y_test.shape))
        #print("batch size: "+str(batch_size))
        #print("fitting now...")
        #print(y_test)
        scores = model.fit(x=x_train, y=y_train, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
        #print("scores: )
        #print("model fitted")
        time_elapsed = time_elapsed + time.time() - t0
        print ("Time Elapsed: " +str(time_elapsed))
        sys.stdout.flush()

        score_train = model.evaluate(x_train, y_train, verbose=0)
        print('Train Loss:', score_train[0])
        print('Train Accuracy:', score_train[1])
        f_train.write(str(score_train)+"\n")

        score_test = model.evaluate(x_test, y_test, verbose=0)
        print('Test Loss:', score_test[0])
        print('Test Accuracy:', score_test[1])
        f_test.write(str(score_test)+"\n")

        f_scores.write(str(score_train[0])+","+str(score_train[1])+","+str(score_test[0])+","+str(score_test[1]) + "\n")

        if SAVE_WEIGHTS and epoch % 5 == 0:
            model.save_weights(weights_path+ "epoch_" + str(epoch) + ".h5")
            print("Saved model to disk in: " + weights_path + model_name + "_epoch" + str(epoch) + ".h5")

    f_train.close()
    f_test.close()
    f_scores.close()

    # Save time elapsed
    f = open(model_path+"time_elapsed.txt", 'w')
    f.write(str(time_elapsed))
    f.close()

    if SAVE_MODEL:
        json_string = model.to_json()
        f = open(model_path+".json", 'w')
        f.write(json_string)
        f.close()

    '''
    # Save files when an sudden close happens / ctrl C
    except:
        f_train.close()
        f_test.close()
        f_scores.close()
        # Save time elapsed
        f = open(model_path+ "time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()
    finally:
        f_train.close()
        f_test.close()
        f_scores.close()
        # Save time elapsed
        f = open(model_path+ "time_elapsed.txt", 'w')
        f.write(str(time_elapsed))
        f.close()
    '''

if TEST:
    t0 = time.time()
    print('Predicting...','\n')

    #real_labels_mean = load_gt(test_gt_list)
    real_labels = y_test

    results = np.zeros((x_test.shape[0], tags.shape[0]))
    #predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
    means = np.zeros((y_test.shape[0], 1))

    n = 0
    incorrect = 0
    print("n_test = "+str(n_test))
    for i in range(0, n_test):
        clause = x_test[i,:,:,:]
        clause = clause[np.newaxis,:,:,:]
        #print("Dimensions of array being tested: "+str(clause.shape))
        results[i] = model.predict(clause, batch_size=None)
        #weights=None, input_tensor=clause,
        #results[i] = model(clause, training=False)
        n = n + 1

        prediction_tensor = results[i]
        if i < 5:
            print("Resulting label (tensor):" + str(prediction_tensor))
        # = tags[predicted_label]
        mean = prediction_tensor#.mean(0)
        #print("mean: "+str(mean))
        sort_result(tags, mean.tolist())
        #print("mean (sorted): "+str(mean))
        final = predict_label(mean)
        means[i] = final
        prediction_array = prediction_tensor
        final = get_label(prediction_array, 0)
        print("final: "+str(final))
        print("real_labels[i]: "+str(real_labels[i]))
        real_label = get_label(real_labels[i], 0)
        #print("final: "+str(final))
        #print("Predicted Genre: "+str(genre))

        if final != real_label:
            incorrect = incorrect + 1
            #print("Incorrect!")

    percent_correct = (n - incorrect) / n
    print("Correct Prediction Rate: "+str(percent_correct)+"%")

    '''cnf_matrix_frames = confusion_matrix(real_labels_frames, predicted_labels_frames)
    plot_confusion_matrix(cnf_matrix_frames, classes=tags, title='Confusion matrix (frames)')

    cnf_matrix_mean = confusion_matrix(real_labels_mean, predicted_labels_mean)
    plot_confusion_matrix(cnf_matrix_mean, classes=tags, title='Confusion matrix (using mean)')
    '''
