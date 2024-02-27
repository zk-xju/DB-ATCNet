# %%
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.manifold import TSNE

import models
from Physionet_DataLoad import load_physionet


# %%
def draw_learning_curves(history, results_path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/ACC_' + '.png')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.savefig(results_path + '/LOSS_' + '.png')
    plt.show()
    plt.close()


def draw_confusion_matrix(cf_matrix, results_path):
    # Generate confusion matrix plot
    # display_labels = ['Left fist','Right fist']#2 class
    display_labels = ['Both feet', 'Left fist', 'Both fists', 'Right fist']#4 class

    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,
                                  display_labels=display_labels)
    disp.plot()
    disp.ax_.set_xticklabels(display_labels, rotation=12)
    plt.title('Confusion Matrix ' )
    plt.savefig(results_path + '/subject_'+ '.png')
    plt.show()


def draw_performance_barChart(metric, label, results_path):
    fig, ax = plt.subplots()
    x = list(range(1, 1))
    ax.bar(x, metric, 0.5, label=label)
    ax.set_ylabel(label)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_title('Model ' + label + ' per subject')
    ax.set_ylim([0, 1])
    plt.savefig(results_path + '/' + label + '.png')

# %% Training
def train(dataset_conf, train_conf, results_path):
    # Get the current 'IN' time to calculate the overall training time
    in_exp = time.time()
    # Create a file to store the path of the best model among several runs
    best_models = open(results_path + "/best models.txt", "w")
    # Create a file to store performance during training
    log_write = open(results_path + "/log.txt", "w")
    # Create a .npz file (zipped archive) to store the accuracy and kappa metrics
    # for all runs (to calculate average accuracy/kappa over all runs)
    perf_allRuns = open(results_path + "/perf_allRuns.npz", 'wb')

    # Get dataset paramters
    data_path = dataset_conf.get('data_path')
    # Get training hyperparamters
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')  # Plot Learning Curves?
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')
    print('Training model: ', model_name)

    # Get the current 'IN' time to calculate the subject training time
    in_sub = time.time()
    # Initiating variables to save the best subject accuracy among multiple runs.
    BestSubjAcc = 0
    bestTrainingHistory = []
    # Get training and test data
    X_train, y_train_onehot,X_test, y_test_onehot = load_physionet(data_path)

    # Get the current 'IN' time to calculate the 'run' training time
    in_run = time.time()
    # Create folders and files to save trained models for all runs
    filepath = results_path + '/saved models/run-{}'.format(1)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = filepath + '/subject-{}.h5'.format(1)

    # Create the model
    model = getModel(model_name)
    # Compile and train the model
    model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                        save_best_only=True, save_weights_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
    ]
    history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Evaluate the performance of the trained model.
    # Here we load the Trained weights from the file saved in the hard
    # disk, which should be the same as the weights of the current model.
    model.load_weights(filepath)
    y_pred = model.predict(X_test).argmax(axis=-1)
    labels = y_test_onehot.argmax(axis=-1)
    acc= accuracy_score(labels, y_pred)
    kappa= cohen_kappa_score(labels, y_pred)

    # Get the current 'OUT' time to calculate the 'run' training time
    out_run = time.time()
    # Print & write performance measures for each run
    info = 'Subject: {}   Train no. {}   Time: {:.3f} m   '.format(1, 1,
                                                                   ((out_run - in_run) / 60))
    info = info + 'Test_acc: {:.5f}   Test_kappa: {:.5f}'.format(acc, kappa)
    print(info)
    log_write.write(info + '\n')
    # If current training run is better than previous runs, save the history.
    if (BestSubjAcc < acc):
        BestSubjAcc = acc
        bestTrainingHistory = history

    # Store the path of the best model among several runs
    best_run = np.argmax(acc)
    filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run + 1, 1) + '\n'
    best_models.write(filepath)
    # Get the current 'OUT' time to calculate the subject training time
    out_sub = time.time()
    # Print & write the best subject performance among multiple runs
    info = '----------\n'
    info = info + 'Subject: {}   best_run: {}   Time: {:.3f} m   '.format(1, best_run + 1,
                                                                          ((out_sub - in_sub) / 60))
    info = info + 'acc: {:.5f}   avg_acc: {:.5f} +- {:.5f}   '.format(acc, np.average(acc),
                                                                      acc.std())
    info = info + 'kappa: {:.5f}   avg_kappa: {:.5f} +- {:.5f}'.format(kappa,
                                                                       np.average(kappa),
                                                                       kappa.std())
    info = info + '\n----------'
    print(info)
    log_write.write(info + '\n')
    # Plot Learning curves
    if (LearnCurves == True):
        print('Plot Learning Curves ....... ')
        draw_learning_curves(bestTrainingHistory, results_path)

    # Get the current 'OUT' time to calculate the overall training time
    out_exp = time.time()
    info = '\nTime: {:.3f} h   '.format((out_exp - in_exp) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    # Store the accuracy and kappa metrics as arrays for all runs into a .npz
    # file format, which is an uncompressed zipped archive, to calculate average
    # accuracy/kappa over all runs.
    np.savez(perf_allRuns, acc=acc, kappa=kappa)

    # Close open files
    best_models.close()
    log_write.close()
    perf_allRuns.close()

    # Evaluation
    model = getModel(model_name)
    test(model, dataset_conf, results_path, X_test, y_test_onehot)

# %% Evaluation
def test(model, dataset_conf, results_path,X_test,y_test_onehot, allRuns=True):
    # Open the  "Log" file to write the evaluation results
    log_write = open(results_path + "/log.txt", "a")
    # Open the file that stores the path of the best models among several random runs.
    best_models = open(results_path + "/best models.txt", "r")

    # Get dataset paramters
    n_classes = dataset_conf.get('n_classes')

    # Initialize variables
    cf_matrix = np.zeros([n_classes, n_classes])

    # Calculate the average performance (average accuracy and K-score) for
    # all runs (experiments) for each subject.
    if (allRuns):
        # Load the test accuracy and kappa metrics as arrays for all runs from a .npz
        # file format, which is an uncompressed zipped archive, to calculate average
        # accuracy/kappa over all runs.
        perf_allRuns = open(results_path + "/perf_allRuns.npz", 'rb')
        perf_arrays = np.load(perf_allRuns)
        acc_allRuns = perf_arrays['acc']
        kappa_allRuns = perf_arrays['kappa']

        # Load data
        X_test, y_test_onehot = X_test, y_test_onehot

        # Load the best model out of multiple random runs (experiments).
        filepath = best_models.readline()
        model.load_weights(results_path + filepath[:-1])

        # Predict MI task
        y_pred = model.predict(X_test).argmax(axis=-1)

        # Calculate accuracy and K-score
        labels = y_test_onehot.argmax(axis=-1)
        acc_bestRun = accuracy_score(labels, y_pred)
        kappa_bestRun = cohen_kappa_score(labels, y_pred)
        # Calculate and draw confusion matrix
        cf_matrix[:, :] = confusion_matrix(labels, y_pred, normalize='pred')
        draw_confusion_matrix(cf_matrix[ :, :], results_path)

        # Print & write performance measures for each subject
        info = 'Subject: {}   best_run: {:2}  '.format(1,
                                                       (filepath[filepath.find('run-') + 4:filepath.find('/sub')]))
        info = info + 'acc: {:.5f}   kappa: {:.5f}   '.format(acc_bestRun, kappa_bestRun)
        if (allRuns):
            info = info + 'avg_acc: {:.5f} +- {:.5f}   avg_kappa: {:.5f} +- {:.5f}'.format(
                np.average(acc_allRuns), acc_allRuns.std(),
                np.average(kappa_allRuns), kappa_allRuns.std())
        print(info)
        log_write.write('\n' + info)

        # t-sne visualization
        y_pred1 = model.predict(X_test)
        tsne = TSNE(n_components=2, random_state=33, verbose=1, n_iter=1000)
        tsne_results2 = tsne.fit_transform(y_pred1)
        color_map = np.argmax(y_test_onehot, axis=1)
        print(color_map.shape)
        plt.figure(figsize=(8, 8))
        display_labels = ['Left fist', 'Right fist']
        for cl in range(2):#2 class
        # for cl in range(4):#4 class
            indices = np.where(color_map == cl)
            print(indices[0].shape)
            indices = indices[0]
            plt.scatter(tsne_results2[indices, 0], tsne_results2[indices, 1], s=8, label=display_labels[cl])
        plt.legend()
        plt.show()
        plt.savefig(results_path + '/t-sne_allsubject' + '.png')

    # Print & write the average performance measures for all subjects
    info = '\nAverage of {} subjects - best runs:\nAccuracy = {:.5f}   Kappa = {:.5f}\n'.format(1,
        np.average(acc_bestRun), np.average(kappa_bestRun))

    print(info)
    log_write.write(info)

    # Close open files
    log_write.close()


# %%
def getModel(model_name):
    # Select the model
    if (model_name == 'DB_ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.DB_ATCNet(
            # Dataset parameters
            n_classes=4,
            in_chans=64,
            in_samples=640,

            # Attention Dual-branch Convolution block (ADBC) parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            drop1=0.35,
            depth1=2,
            depth2=4,

            # Sliding window (SW) parameter
            n_windows=5,

            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'

            # Temporal convolutional Fusion Network block (TCFN) parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            drop2=0.1,
            drop3=0.15,
            drop4=0.15,

            tcn_activation='elu',
        )
    elif (model_name == 'ATCNet'):
        # Train using the proposed model (ATCNet): https://doi.org/10.1109/TII.2022.3197419
        model = models.ATCNet(
            # Dataset parameters
            n_classes=4,
            in_chans=64,
            in_samples=640,
            # Sliding window (SW) parameter
            n_windows=5,
            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
    elif (model_name == 'TCNet_Fusion'):
        # Train using TCNet_Fusion: https://doi.org/10.1016/j.bspc.2021.102826
        model = models.TCNet_Fusion(n_classes=4)
    elif (model_name == 'EEGTCNet'):
        # Train using EEGTCNet: https://arxiv.org/abs/2006.00622
        model = models.EEGTCNet(n_classes=4)
    elif (model_name == 'EEGNet'):
        # Train using EEGNet: https://arxiv.org/abs/1611.08024
        model = models.EEGNet_classifier(n_classes=4)
    elif (model_name == 'EEGNeX'):
        # Train using EEGNeX: https://arxiv.org/abs/2207.12369
        model = models.EEGNeX_8_32(n_timesteps=640, n_features=64, n_outputs=4)
    elif (model_name == 'DeepConvNet'):
        # Train using DeepConvNet: https://doi.org/10.1002/hbm.23730
        model = models.DeepConvNet(nb_classes=4, Chans=64, Samples=640)
    elif (model_name == 'ShallowConvNet'):
        # Train using ShallowConvNet: https://doi.org/10.1002/hbm.23730
        model = models.ShallowConvNet(nb_classes=4, Chans=64, Samples=640)
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))

    return model

# %%
def run():
    # Get dataset path
    data_path = "/root/autodl-tmp/physionet"

    # Create a folder to store the results of the experiment
    results_path = os.getcwd() + "/results"
    if not os.path.exists(results_path):
        os.makedirs(results_path)  # Create a new directory if it does not exist

    # 10-fold cross-validation
    # for i in range(10):
        # Set dataset paramters
        dataset_conf = {'n_classes': 4, 'n_channels': 64, 'data_path': data_path}
        # Set training hyperparamters
        train_conf = {'batch_size': 32, 'epochs': 500, 'patience': 100, 'lr': 0.0009,
                      'LearnCurves': True, 'model': 'DB_ATCNet'}

        model = getModel(train_conf.get('model'))
        model.summary()

        # Train the model
        train(dataset_conf, train_conf, results_path)

        # Evaluate the model based on the weights saved in the '/results' folder
        # The test function is called at the end of the train function

# %%
if __name__ == "__main__":
    run()