import numpy as np
import tensorflow as tf
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
import os
from imagesUtil import getThreeTransformationsImages
import random
from datetime import datetime
from vis.visualization import visualize_saliency


train=True


class create_Video():
    """this class helps to store data as image and later as video
    """
    def __init__(self):
        self.videos=dict()
        self.imageCount = 0

    def create_Videos(self):
        """
        creates the Videos from all stored frames after the training
        """
        for vid in self.videos:
            height, width, layers = self.videos[vid][0].shape
            size = (width,height)
            video_name=vid+".avi"
            out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            for i in range(len(self.videos[vid])):
                out.write(self.videos[vid][i])
            out.release()

    def store_Frame_to_Video(self,fig, name_of_video="name"):
        """
        stores figures into a list
        fig: figure to store
        name_of_video: name of the video
        """
        #store the figure(frames) as image
        fig.canvas.draw()
        img=np.frombuffer(fig.canvas.tostring_rgb(),dtype=np.uint8)
        img=img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        img=cv2.cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        #add to list of queue for images (to receive video in the end wwith help of create_Videos function)
        if name_of_video not in self.videos:
            self.videos[name_of_video]=[]
            print("created list")
        #print(len(self.videos[name_of_video]))
        self.videos[name_of_video].append(img)


def plot_input_layer_FilterWeigths(model,x_test,y_test):
    video_store.imageCount += 1
    if video_store.imageCount == 20:
        # retrieve weights from the second hidden layer
        filters, biases = model.layers[0].get_weights()
        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters

        n_filters = 32
        fig, axis = plt.subplots(6,16,figsize=(8,3))
        for i in range(3):
            for j in range(32):
                filt = filters[:,:,:,j]
                if j > 15:
                    axis[2*i+1][j-16].set_yticks([])
                    axis[2*i+1][j-16].set_xticks([])
                    axis[2*i+1][j-16].imshow(filt[:,:,i],cmap='gray')
                else:
                    axis[2*i][j].set_yticks([])
                    axis[2*i][j].set_xticks([])
                    axis[2*i][j].imshow(filt[:,:,i],cmap='gray')

        #store the filters as image
        video_store.store_Frame_to_Video(fig,"Filter")
        # show the figure
        plt.show()
        #free memory
        video_store.imageCount = 0
        plt.close(fig)


def plot_input_layer_FeatureMaps(model,x_test,y_test):
    video_store.imageCount += 1
    if video_store.imageCount == 20:


        #get wanted outputs
        layer_outputs = model.layers[9].output#[layer.output for layer in model.layers[:2]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

        activations = activation_model.predict(x_test[:4]) # Returns a list of five Numpy arrays: one array per layer activation

        #get a layer activation for a picture
        first_layer_activation = activations[0]
        #print(first_layer_activation.shape)

        #plot every neuron activation
        fig, axs = plt.subplots(4, 8, figsize=(8, 4), facecolor='w', edgecolor='k')
        #fig.subplots_adjust(hspace = .5, wspace=.001)
        axs = axs.ravel()

        for i in range(32):
            axs[i].matshow(first_layer_activation[:, :, i], cmap='viridis')
            axs[i].set_yticks([])
            axs[i].set_xticks([])
        video_store.store_Frame_to_Video(fig,"featuremaps")
        video_store.imageCount = 0
        #plt.show()
        plt.close(fig)

def get_transformed_Images(x_test):
    '''
    Arguments: takes the test image array

    returns: returns an array with 3*9 images for base image 1
             and 3*9 for image 2 and 3*9 for image 3
    '''
    #create the array with the transformed pictures
    pictures = []
    for x in range(3):
        (h, w) = x_test[x].shape[:2]
        #append the base picture for rotation
        pictures.append(x_test[x])
        #then append the 8 rotations
        for i in range(8):
            center = (w / 2, h / 2)

            M = cv2.getRotationMatrix2D(center, (i+1)*45, 1.0)
            rotated = cv2.warpAffine(x_test[x], M, (w, h))
            pictures.append(rotated)
        #append base picture for zooming
        pictures.append(x_test[x])
        #append the 8 zoom pictures
        for i in range(8):
            cropped = x_test[x,(i+1):h-(i+1), (i+1):w-(i+1)]
            cropped = cv2.resize(cropped, (w, h), interpolation = cv2.INTER_AREA)
            pictures.append(cropped)
        #append the base picture and the additional 8 for movement
        for i in range(9):
            cropped = x_test[x,(i):(i+24), 3:28]
            cropped = cv2.resize(cropped, (w, h), interpolation = cv2.INTER_AREA)
            pictures.append(cropped)

    #convert the list into an array
    pictures = np.asarray(pictures)

    #plot the pictures
    #fig, axs = plt.subplots(9,9,figsize=(10,10))
    #axs = axs.ravel()

    #for i in range(81):
    #    axs[i].imshow(pictures[i], cmap='hsv',interpolation='none')
    #    axs[i].set_yticks([])
    #    axs[i].set_xticks([])
    #plt.show()
    return pictures

def plot_Denselayer_Euclidean_Distances(model,transformedImages):
    #video_store.imageCount += 1
    if video_store.imageCount == 0:
        fig, axs = plt.subplots(4,3, figsize=(20,10))
        layerNames = ["dense_one", "dense_two", "dense_three", "dense_four"]
        for x in range(4):
            layer_outputs = model.get_layer(layerNames[x])
            activation_model = models.Model(inputs=model.input, outputs=layer_outputs.output)
            activations = activation_model.predict(transformedImages)
            rotationDistance = []
            zoomDistance = []
            movementDistance = []
            for i in range (9):
                rotationDistance.append(np.linalg.norm(activations[0]-activations[i+0]))
                zoomDistance.append(np.linalg.norm(activations[9]-activations[i+9]))
                movementDistance.append(np.linalg.norm(activations[18]-activations[i+18]))

            for i in range (9):
                rotationDistance.append(np.linalg.norm(activations[27]-activations[i+27]))
                zoomDistance.append(np.linalg.norm(activations[36]-activations[i+36]))
                movementDistance.append(np.linalg.norm(activations[45]-activations[i+45]))

            for i in range (9):
                rotationDistance.append(np.linalg.norm(activations[54]-activations[i+54]))
                zoomDistance.append(np.linalg.norm(activations[63]-activations[i+63]))
                movementDistance.append(np.linalg.norm(activations[72]-activations[i+72]))

            axs[x][0].plot(rotationDistance[0:9], label='Picture 1')
            axs[x][0].plot(rotationDistance[9:18], label='Picture 2')
            axs[x][0].plot(rotationDistance[18:27], label='Picture 3')
            #axs[x][0].legend()
            axs[x][0].set_title('Denselayer = '+ str(x)+' , Transormation = Rotation')
            if x < 3:
                axs[x][0].set_xticks([])
            axs[3][0].set_xticklabels(['-45','0', '45', '90', '135', '180', '225', '270', '315', '360'])

            axs[x][1].plot(zoomDistance[0:9], label='Picture 1')
            axs[x][1].plot(zoomDistance[9:18], label='Picture 2')
            axs[x][1].plot(zoomDistance[18:27], label='Picture 3')
            #axs[x][1].legend()
            axs[x][1].set_title('Denselayer = '+ str(x)+' , Transormation = Scaling')
            if x < 3:
                axs[x][1].set_xticks([])

            axs[x][2].plot(movementDistance[0:9], label='Picture 1')
            axs[x][2].plot(movementDistance[9:18], label='Picture 2')
            axs[x][2].plot(movementDistance[18:27], label='Picture 3')
            #axs[x][2].legend()
            axs[x][2].set_title('Denselayer = '+ str(x)+' , Transormation = Moving')
            if x < 3:
                axs[x][2].set_xticks([])


        video_store.store_Frame_to_Video(fig,"distances")
        video_store.imageCount = 0
        #plt.show()
        plt.close(fig)

class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        x_train, y_train, x_test, y_test, class_names = get_dataset( 'cifar10', used_labels, training_size, test_size )

        #pick a random image for the calculation of the sensitivity maps
        #this is done after every epoch
        self.imageIndex = random.randint(0, len(x_test))
        self.obs_input = x_test[self.imageIndex]

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
    #     """This function is executed after a new batch begin in this case after 32 trials
    #     batch: contains information about the batch
    #     logs: logs are stored here if wanted, currently batchsize and iteration number of batch are stored
    #     """
    
        #plot_input_layer_FilterWeigths(model,x_test,y_test)
        plot_input_layer_FeatureMaps(model,x_test,y_test)

    def on_batch_begin(self,batch,logs={}):
        pass


    #performs the smoothGrad routine.
    #unfortunately, this takes quite a lot of time
    #using fewer reps speeds up the process drastically but makes the smoothening less effective
    def on_epoch_end(self, epoch, logs={}):

        enableSmoothGrad = True

        if enableSmoothGrad:
        
            #print information about the input image
            #these are not necessary but can be helpful
            if epoch == 0:
                plt.imshow(self.obs_input.astype('uint8'))
                plt.savefig('inputImage.png')
                print('\n')
                print(class_names[y_test[self.imageIndex]])
                print('image index = ' + str(self.imageIndex))

            #this determines how many noisy images are used for the smoothening process
            reps = 25

            #the layer to visualize
            vis_layer = 20

            #initial noise level
            noiseLevel = 0

            print('\nCalculating smoothened sensitivity maps...')
            while noiseLevel <= 0.5:

                print('Smoothening sensitivityMap using noiseLevel ' + str(noiseLevel) + '... ')

                #get highest and lowest values for each channel
                x_max_ch0 = np.max(self.obs_input[:,:,0])
                x_min_ch0 = np.min(self.obs_input[:,:,0])
                x_max_ch1 = np.max(self.obs_input[:,:,1])
                x_min_ch1 = np.min(self.obs_input[:,:,1])
                x_max_ch2 = np.max(self.obs_input[:,:,2])
                x_min_ch2 = np.min(self.obs_input[:,:,2])

                #for every channel, calculate the standard deviation that results in the desired noise level
                stdDev_ch0 = noiseLevel * (x_max_ch0 - x_min_ch0)
                stdDev_ch1 = noiseLevel * (x_max_ch1 - x_min_ch1)
                stdDev_ch2 = noiseLevel * (x_max_ch2 - x_min_ch2)
                    

                try:
                    os.mkdir(str(self.imageIndex))
                except:
                    pass

                try:
                    os.mkdir(str(self.imageIndex)+'/'+str(noiseLevel))
                except:
                    pass
                
                
                if noiseLevel != 0:
                    #create an array which holds the sum of all sensitivity maps calculated 
                    accumulator = np.zeros((32,32))
                    for rep in range(0, reps):
                        random.seed(datetime.now())
                        
                        row,col,ch = self.obs_input.shape
                        mean = 0

                        #create gaussian noise for every color channel 
                        gauss_ch0 = np.random.normal(mean,stdDev_ch0,(row,col))
                        gauss_ch1 = np.random.normal(mean,stdDev_ch1,(row,col))
                        gauss_ch2 = np.random.normal(mean,stdDev_ch2,(row,col))

                        #combine noise arrays
                        gauss = np.dstack((gauss_ch0,gauss_ch1,gauss_ch2))

                        #add the noise to the input image
                        noisy = self.obs_input + gauss

                        #add the sensitivity map to the storage array
                        accumulator += visualize_saliency(model, vis_layer, seed_input=noisy, filter_indices=None)


                    #calculate the average
                    accumulator /= reps

                    #save the sensitivity map
                    plt.imshow(accumulator, cmap='gray')
                    plt.savefig(str(self.imageIndex)+'/'+str(noiseLevel)+'/'+str(epoch)+'saliencyMap_smooth.png')
                
                #for better comparison, save the input image in the respective folders
                if epoch == 0:
                    plt.imshow(self.obs_input.astype('uint8'))
                    plt.savefig(str(self.imageIndex)+'/'+str(noiseLevel)+'/inputImage.png')

                #save the unsmoothened sensitivity map
                originalSaliencyMap = visualize_saliency(model, vis_layer, seed_input=self.obs_input, filter_indices=None)
                plt.imshow(originalSaliencyMap, cmap='gray')
                plt.savefig(str(self.imageIndex)+'/'+str(noiseLevel)+'/'+str(epoch)+'saliencyMap.png')
                noiseLevel += 0.1
        else:
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



'''-----------------------------------------------------------------------------'''
    

if __name__ == '__main__':

    trainings = 1
    for training in range(0, trainings):
        with tf.Session() as sess:
            random.seed(datetime.now())


            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())


            #create Video storage class
            video_store=create_Video()

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
            transformedImages = get_transformed_Images(x_test)
            a, b ,c = getThreeTransformationsImages(transformedImages)
            #plot the pictures
            """
            fig, axs = plt.subplots(3,figsize=(10,10))
            axs = axs.ravel()

            axs[0].imshow(a,interpolation='none')
            axs[0].set_yticks([])
            axs[0].set_xticks([])
            axs[1].imshow(b, cmap='hsv')
            axs[1].set_yticks([])
            axs[1].set_xticks([])
            axs[2].imshow(c, cmap='hsv')
            axs[2].set_yticks([])
            axs[2].set_xticks([])
            #plt.show()
            """

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
                model.add(Dense(512, activation='relu', name="dense_one"))
                model.add(Dropout(0.25))
                model.add(Dense(512, activation='relu', name="dense_two"))
                model.add(Dropout(0.25))
                model.add(Dense(512, activation='relu', name="dense_three"))
                model.add(Dropout(0.25))
                model.add(Dense(512, activation='relu', name="dense_four"))
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
                model = load_model('cifar10Model0.6043.h5', custom_objects=None, compile=True)
                #data= np.load('cifar10Model0.6408.npy',allow_pickle=True).item()

            print(model.summary())
            video_store.create_Videos()
            #plot_input_layer_FilterWeigths(model,x_test,y_test)
