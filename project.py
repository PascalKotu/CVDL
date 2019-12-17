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

train=True
data=dict()
data["Dense"]=[]

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    # def on_batch_begin(self,batch,logs={}):
        
    #     """This function is executed before a new batch begin in this case after 32 trials
    #     batch: contains information about the batch
    #     logs: logs are stored here if wanted, currently batchsize and iteration number of batch are stored
    #     """
        
    #     trial=logs["batch"]*logs["size"]
    #     #save the last layer activation (bis jetzt nur von plot_heatmap und plot_output_layer gebraucht)
    #     #inp=model.input
    #     outputs=[]
    #     layer_outputs=model.layers[13].output
    #     activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    #     #predicts = model.predict(x_test[:4])
    #     activations = activation_model.predict(x_test[:4])
    #     first_layer_activation = activations[0]
    #     print(first_layer_activation.shape)

    #     #plot every neuron activation
    #     fig, axs = plt.subplots(4,8, figsize=(15, 6), facecolor='w', edgecolor='k')
    #     fig.subplots_adjust(hspace = .5, wspace=.001)

    #     axs = axs.ravel()

    #     for i in range(32):
    #         axs[i].matshow(first_layer_activation[:, :, i], cmap='viridis')
    #     plt.show()
        
        
        #output_tmp=model.layers[13].output[-2:]
        #print(output_tmp)
        #data["Dense"].append(output_tmp)
        ##print(output_tmp.shape)
        #np.save("t.npy",data["Dense"])
        #input("sds")
        """for i, layer in enumerate(model.layers):
            if i > 0:
                outputs.append(layer.output)
        functor = backend.function([inp]+[backend.learning_phase()],outputs)
        layer_activation=dict()
        for name in class_names:
            observation=cv2.resize(x_test[trial],dsize=(32,32))
            layer_activation[name]=functor([[[observation]],1.])[-2:]"""
        #for color in rlAgent.OAIInterface.modules['worldModule'].imagesSymbols:
            #layer_activation[color]=dict()
            #for stim in rlAgent.experimentalDesign["Stimulus"]:
                #observation=cv2.resize(rlAgent.OAIInterface.modules['worldModule'].imagesSymbols[color][rlAgent.experimentalDesign["Stimulus"][stim].split(".")[0]],dsize=(32,32))
                #observation.astype('float32')
                #observation=observation/255.0
                #layer_activation[color][stim]=functor([[[observation]],1.])[-2:]
        #layer_activation=functor([[[observation]],1.])[-2:]
        #rlAgent.OAIInterface.data.append(layer_activation)

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

    np.save('cifar10Model'+str(test_scores[1])+'.npy',data)
    model.save('cifar10Model'+str(test_scores[1])+'.h5')
    print("Finished training with {:4.1f}% training and {:4.1f}% testing accuracy"
            .format(train_scores[1] * 100, test_scores[1] * 100))

else:
    #load model
    model = load_model('cifar10Model0.6408.h5', custom_objects=None, compile=True)
    #data= np.load('cifar10Model0.6408.npy',allow_pickle=True).item()


def plot_input_layer(model,x_test,y_test):
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


def plot_heatmap(model, class_names):
    """this function plots the neuronal activation of the Dense layer
    model: contains network architecture and its learned parameters
    """
    activation_map=dict()
    for name in class_names:
        activation_map[name]=[]
    activation_map["frog"]=data["frog"]

    plt.imshow(activation_map["frog"], cmap='plasma', interpolation='nearest')
    plt.show()

    """#go through the session
    while d[i]["session"]==session_wanted:
        def context_feedback(context):
            activation_map["control_left"].append(d[i][context]["control_left"][-2][0])
            activation_map["control_right"].append(d[i][context]["control_right"][-2][0])
            activation_map["novel_left"].append(d[i][context]["novel_left"][-2][0])
            activation_map["novel_right"].append(d[i][context]["novel_right"][-2][0])

        if d[i]["activePhase"]=="acquisition":
            context_feedback(experimentalDesign["context"][0])
        elif d[i]["activePhase"]=="extinction":
            context_feedback(experimentalDesign["context"][1])
        else:
            context_feedback(experimentalDesign["context"][2])

        #when the last trial of the experiment is reached end this loop
        if d[i]["index"]==d[-1]["index"]:
            break
        i+=1

    for key in activation_map.keys():
        activation_map[key]=np.swapaxes(np.array(activation_map[key]),0,1)

    fig=plt.figure(figsize=(20,8))
    plt.suptitle("Q-activation")

    #plot the activation map for all 4 Stimuli in this Session
    for i,stim in enumerate(activation_map.keys()):
        plt.subplot("41%i"%i)
        plt.title(stim)
        plt.imshow(activation_map[stim], cmap='plasma', interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.yticks(np.arange(0,64,25),[0,25,50])
        plt.xscale('linear')
        plt.ylabel("neuron")
        plt.xlabel("trial")
        plt.colorbar()
    plt.tight_layout()

    if save_folder is not None:
        plt.savefig(save_folder+"/qvalue_heatmap_session_%i.png"%session_wanted,dpi=200)
        plt.close()
    else:
        plt.show()"""


def plot_output_layer(model):
    """this function plots the neuronal activation of the output layer
    model: contains network architecture and its learned parameters
    """
    d=data["data"]
    #iterate to the session we want the output from
    i=0
    while d[i]["session"]!=session_wanted:
        i+=1

    activation={"control_left":[],"control_right":[],"novel_left":[],"novel_right":[]}

    acquisition_len=0
    extinction_len=0

    #go through the session
    while d[i]["session"]==session_wanted:
        #get the lenght of the Phases to plot a line between them
        if d[i]["activePhase"]=="acquisition":
            acquisition_len+=1
        if d[i]["activePhase"]=="extinction":
            extinction_len+=1

        #iterate over the data to get the output-layer activation for each stimuli and the right Phase
        def context_feedback(context):
            activation["control_left"].append(d[i][context]["control_left"][-1][0])
            activation["control_right"].append(d[i][context]["control_right"][-1][0])
            activation["novel_left"].append(d[i][context]["novel_left"][-1][0])
            activation["novel_right"].append(d[i][context]["novel_right"][-1][0])
        
        if d[i]["activePhase"]=="acquisition":
            context_feedback(experimentalDesign["context"][0])
        elif d[i]["activePhase"]=="extinction":
            context_feedback(experimentalDesign["context"][1])
        else:
            context_feedback(experimentalDesign["context"][2])

        #when the last trial of the experiment is reached end this loop
        if d[i]["index"]==d[-1]["index"]:
            break
        i+=1


    fig=plt.figure(figsize=(20,8))
    #plt.title("Q-activation-Output")

    for i,stim in enumerate(activation.keys()):
        plt.subplot("41%i"%i)
        plt.title(stim)
        plt.plot(activation[stim])
        plt.plot([acquisition_len,acquisition_len],[-1.5,1.5],'black')
        plt.plot([acquisition_len+extinction_len,acquisition_len+extinction_len],[-1.5,1.5],'black')
        plt.xlabel("trial")
        plt.ylabel("neuron")
        plt.ylim(-1.5,1.5)
    plt.tight_layout()


    #save the plot
    if save_folder is not None:
        plt.savefig(save_folder+"/qvalue_output_session_%i.png"%(session_wanted),dpi=200)
        plt.close()
    else:
        plt.show()





plot_input_layer(model,x_test,y_test)
#plot_heatmap(model,class_names)
#plot_output_layer()

