import keras
import numpy as np
def get_lenet(input_shape, output_size = 10):
    #Creates Sequential model using Keras
    #Number of nodes is the same as number of features (different number of nodes were tried but it did not
    #affect validation accuracy significantly)
    network = keras.Sequential([
                                #Input layer:
                                keras.layers.Conv2D(20, 5, padding="same", input_shape=input_shape, use_bias=True),
                                #Hidden Layers:
                                keras.layers.Activation(activation="relu"),
                                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                                keras.layers.Conv2D(50, 5, padding="same"),
                                keras.layers.Activation(activation="relu"),
                                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
                                keras.layers.Flatten(),
                                keras.layers.Dense(500),
                                keras.layers.Activation(activation="relu"),
                                keras.layers.Dense(output_size, name='vis',use_bias=True),
                                #Output layer
                                keras.layers.Activation(activation="softmax"),
                            ])
    return(network)

# Clones a neural network architecture into an ensemble with randomly initialised weights (structured as a list)
def clone_network_into_ensemble(number_of_networks, network):
    ensemble = []
    for i in range(0,number_of_networks):
        ensemble.append(keras.models.clone_model(network))
    return ensemble

def train_network(network,trainX,trainY,epochs = 5):
    #Compiles sequential model
    #Using learning rate 0.01
    #Loss function will be categorical crossentropy
    network.compile(
                    optimizer=keras.optimizers.SGD(lr=0.01),
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy']
                    )
    #Trains network over a number of epochs
    network.fit(trainX, trainY, epochs = epochs)

    return network

def train_ensemble(ensemble,trainX,trainY,epochs = 5):
    #Compiles sequential model
    #Using learning rate 0.01
    #Loss function will be categorical crossentropy
    for network in ensemble:
        network.compile(
                        optimizer=keras.optimizers.SGD(lr=0.01),
                        loss = 'categorical_crossentropy',
                        metrics = ['accuracy']
                        )
    #Trains network over a number of epochs and evaluates network agains validation data
    #after each epoch
    for network in ensemble:
        network.fit(trainX, trainY, epochs = epochs)

    return ensemble

#Function to get average output values for each network in the ensemble
def get_ensemble_output_values(ensemble,input,number_of_output_nodes):

    predictions = np.zeros(number_of_output_nodes)

    for network in ensemble:
        predictions = predictions + network.predict(np.expand_dims(input,axis=0))

    prediction_average = predictions / len(ensemble)

    return(prediction_average)

#Function to get average output values from each network in ensemble for multiple inputs
def get_ensemble_output_values_for_multiple_inputs(ensemble,inputs,number_of_output_nodes):

    predictions = np.zeros((np.size(inputs, axis=0), number_of_output_nodes))
    print(np.shape(predictions))

    for network in ensemble:
        predictions = predictions + network.predict(inputs)

    print(np.shape(predictions))
    prediction_average = predictions / len(ensemble)

    return(prediction_average)

def get_ensemble_votes(ensemble,input,number_of_output_nodes):

    votes = np.zeros(number_of_output_nodes)

    for network in ensemble:
        predicted_output = np.argmax(network.predict(np.expand_dims(input,axis=0)))
        votes[predicted_output] = votes[predicted_output] + 1

    return(votes)

def get_ensemble_votes_for_multiple_inputs(ensemble,inputs,number_of_output_nodes):

    votes = np.zeros((np.size(inputs, axis=0), number_of_output_nodes))

    for network in ensemble:
        network_votes = np.argmax(network.predict(inputs),axis = 1)
        for i in range(0,np.size(inputs, axis=0)):
            votes[i][network_votes[i]] = votes[i][network_votes[i]] + 1

    return(votes)
#Function to calculate ensemble output (for one input) using mean of outputs
def get_ensemble_predicted_output(ensemble,input,number_of_output_nodes):

    prediction_average = get_ensemble_output_values(ensemble,input,number_of_output_nodes)
    output = np.argmax(prediction_average)

    return(output)

#Function to calculate ensemble outputs (for series of inputs) using mean of outputs
def get_ensemble_predicted_outputs(ensemble,inputs,number_of_output_nodes):

    prediction_average = get_ensemble_output_values_for_multiple_inputs(ensemble,inputs,number_of_output_nodes)
    outputs = np.argmax(prediction_average, axis=1)

    return(outputs)

def get_ensemble_predicted_output_with_votes(ensemble,input,number_of_output_nodes):

    votes = get_ensemble_votes(ensemble,input,number_of_output_nodes)
    output = np.argmax(votes)

    return(output)

def get_ensemble_predicted_outputs_with_votes(ensemble,inputs,number_of_output_nodes):

    votes = get_ensemble_votes_for_multiple_inputs(ensemble,inputs,number_of_output_nodes)
    outputs = np.argmax(votes,axis=1)

    return(outputs)

def evaluate_ensemble_accuracy(ensemble,testX,testY):

    correct = np.sum(np.equal(get_ensemble_predicted_outputs(ensemble,testX,10),np.argmax(testY,axis = 1)))
    return correct/len(testX)
