from keras import backend as K
import os
import time
import h5py
import sys
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
model = MusicGenre_CNN(input_tensor=(128, 647, 1))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Print summary of the model
model.summary()


# Train model
if TRAIN:
    #try:
    print ("Training the model")
    f_train = open(model_path+"scores_train.txt", 'w')
    f_test = open(model_path+"scores_test.txt", 'w')
    f_scores = open(model_path+"scores.txt", 'w')
    for epoch in range(1,n_epochs+1):
        t0 = time.time()
        print ("Number of epoch: " +str(epoch)+"/"+str(n_epochs))
        sys.stdout.flush()
        scores = model.fit(x=x_train, y=y_train, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test))
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

    f_train.close()
    f_test.close()
    f_scores.close()

    # Save time elapsed
    f = open(model_path+"time_elapsed.txt", 'w')
    f.write(str(time_elapsed))
    f.close()

    
if TEST:
    t0 = time.time()
    print('Predicting...','\n')

    real_labels = y_test

    results = np.zeros((x_test.shape[0], tags.shape[0]))
    means = np.zeros((y_test.shape[0], 1))

    n = 0
    incorrect = 0
    print("n_test = "+str(n_test))
    for i in range(0, n_test):
        clause = x_test[i,:,:,:]
        clause = clause[np.newaxis,:,:,:]
        results[i] = model.predict(clause, batch_size=None)
        n = n + 1

        prediction_tensor = results[i]
        if i < 5:
            print("Resulting label (tensor):" + str(prediction_tensor))
        mean = prediction_tensor
        sort_result(tags, mean.tolist())
        final = predict_label(mean)
        means[i] = final
        prediction_array = prediction_tensor
        final = get_label(prediction_array, 0)
        print("final: "+str(final))
        print("real_labels[i]: "+str(real_labels[i]))
        real_label = get_label(real_labels[i], 0)

        if final != real_label:
            incorrect = incorrect + 1

    percent_correct = (n - incorrect) / n
    print("Correct Prediction Rate: "+str(percent_correct)+"%")

    '''cnf_matrix_frames = confusion_matrix(real_labels_frames, predicted_labels_frames)
    plot_confusion_matrix(cnf_matrix_frames, classes=tags, title='Confusion matrix (frames)')

    cnf_matrix_mean = confusion_matrix(real_labels_mean, predicted_labels_mean)
    plot_confusion_matrix(cnf_matrix_mean, classes=tags, title='Confusion matrix (using mean)')
    '''
