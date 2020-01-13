import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras import optimizers
from DLCVDatasets import get_dataset
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import time
from tensorflow.keras import backend
import cv2
from tensorflow.keras.utils import plot_model

train=False
data=dict()
data["Dense"]=[]

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_batch_begin(self,batch,logs={}):
    #     """This function is executed before a new batch begin in this case after 32 trials
    #     batch: contains information about the batch
    #     logs: logs are stored here if wanted, currently batchsize and iteration number of batch are stored
    #     """
        pass


def normalize_data(x_train, x_test):
    # Convert to float32
    x_train = np.float32(x_train)
    x_test = np.float32(x_test)

    # Remove mean
    mean = np.mean(x_train)
    x_train -= mean
    x_test -= mean

    # Standard deviation of 1
    std = np.std(x_train)
    x_train = x_train / std
    x_test = x_test / std

    return x_train, x_test


###create the network###

used_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training_size = 50000
test_size = 10000
num_classes = len(used_labels)

#get data
x_train, y_train, x_test, y_test, class_names = get_dataset( 'cifar10', used_labels, training_size, test_size )

#normalize data
x_train, x_test = normalize_data(x_train,x_test)
#reshape data to fit model
x_train = x_train.reshape(training_size,32,32,3)
x_test = x_test.reshape(test_size,32,32,3)

#create model
model = Sequential()
if train:
    #add model layers
    model.add(Conv2D(32, (3, 3), padding='same', input_shape = (32, 32, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', input_shape = (32, 32, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', input_shape = (32, 32, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))


    ###train the network or get already learned one from file###

    call = LossHistory()
    #compile model using accuracy to measure model performance
    sgd = optimizers.SGD(lr=0.1, momentum=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #train the model
    model.fit(x_train, y_train, validation_split=0.2, epochs=5, callbacks= [call], verbose = 1)

    #evaluate the accuracy
    train_scores = model.evaluate(x_train, y_train, verbose=2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)

    #np.save('cifar10Model'+str(test_scores[1])+'.npy',data)
    model.save('cifar10Model'+str(test_scores[1])+'.h5')
    print("Finished training with {:4.1f}% training and {:4.1f}% testing accuracy"
            .format(train_scores[1] * 100, test_scores[1] * 100))

else:
    #load model
    model = load_model('cifar10Model0.6172.h5', custom_objects=None, compile=True)
    #data= np.load('cifar10Model0.6408.npy',allow_pickle=True).item()


def plot_input_layer_FeatureMaps(model,x_test,y_test):
    #predict model outputs
    predicts = model.predict(x_test[:4])
    print(np.argmax(predicts[0]), np.argmax(predicts[1]), np.argmax(predicts[2]), np.argmax(predicts[3]))
    print(y_test[:4])

    #get wanted outputs
    layer_outputs = model.layers[0].output#[layer.output for layer in model.layers[:2]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(x_test[:4]) # Returns a list of five Numpy arrays: one array per layer activation

    #get a layer activation for a picture
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)

    #plot every neuron activation
    fig, axs = plt.subplots(4,8, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)

    axs = axs.ravel()

    for i in range(32):
        axs[i].matshow(first_layer_activation[:, :, i], cmap='viridis')
    plt.show()


def plot_input_layer_FilterWeigths(model,x_test,y_test):
    # retrieve weights from the second hidden layer
    filters, biases = model.layers[0].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    print(len(filters))
    # plot first few filters
    n_filters = 32
    fig, axis = plt.subplots(3,32,figsize=(15,15))
    for i in range(3):
        for j in range(32):
            filt = filters[:,:,:,j]
            axis[i][j].set_yticks([])
            axis[i][j].set_xticks([])
            axis[i][j].imshow(filt[:,:,i],cmap='gray')




    # show the figure
    plt.show()




plot_input_layer_FilterWeigths(model,x_test,y_test)
plot_input_layer_FeatureMaps(model,x_test,y_test)
