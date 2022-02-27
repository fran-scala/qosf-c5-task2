#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 12:21:20 2022

@author: francesco
"""

import pennylane as qml # version 0.21.0
from pennylane import numpy as np
import time as t
import matplotlib.pyplot as plt #matplotlib version '3.4.3'


def ParseFile(filename):
    '''
    Parameters
    ----------
    filename : .csv file with data and labels

    Returns
    -------
    dataset : dataset with data features as a np.array
    labels : correct labels as a np.array

    '''
    #open the file splitting it in lines
    fh = open(filename, 'r').read().split('\n')[:-1]#the last element is ''
    
    dataset = []
    labels = []
    
    #first line are column headings
    for line in fh[1:]:
        
        row = line.split(',')
        new_row = list(map(lambda x: float(x), row))
        
        dataset.append(np.array(new_row[0:4], requires_grad=True))#columns 0-3 are the features
        labels.append(new_row[4])#5th column are the labels
      
    #transforming lists into arrays    
    dataset = np.array(dataset, requires_grad=True)
    labels = np.array(labels, dtype = int, requires_grad=True)
        
    return dataset, labels


def construct_circuit(feat, angles, n_layers, embedding='amplitude', rot='X'):
    '''
    Function that encodes the features with the specified embedding and then 
    applies the variational ansatz

    Parameters
    ----------
    feat : np.array with data features to be encoded in the circuit
    angles: np.array with the rotation angles
    n_layers : number of layers of the ansatz
    embedding : type of embedding of the features passed as a string, optional.
             The default is 'amplitude'. The others are 'angle'.
    rot: String specifying the kind of rotations in the embedding. 
        Only required if 'angle' embedding is selected. The default is 'X'. 
        The others are 'Y', 'Z'. 

    Returns
    -------
   Sum of the probabilities corresponding to each state on the 2nd half of
   the computational base

    '''
    #defining the circuit for amplitude encoding
    if embedding=='amplitude':
        
        def add_layer(angles, n_qubits):
            '''
            Function that applies one layer of the variational anstaz with the given 
            angles on a register of n_qubits with only RY rotations and CNOTs
            
            Parameters
            ----------
            angles : np.array with the rotation angles
            n_qubits : number of qubits of the circuit
        
            Returns
            -------
            None.
        
            '''
                
            for i in range(n_qubits):
                qml.RY(angles[i], wires=i)
            
            for i in range(n_qubits-1):
                    qml.CNOT(wires=[i,i+1])
        
        n_qubits = 2

        dev = qml.device('default.qubit', wires=n_qubits)
    
        @qml.qnode(dev)
        def circuit():
            qml.AmplitudeEmbedding(feat, wires=range(n_qubits), normalize=True)
            
            k=0
            for i in range(n_layers):
                add_layer(angles[k:k+1*n_qubits], n_qubits)
                k+=1*n_qubits
            
            return qml.probs(wires = range(n_qubits))
            
    #defining the circuit for angle encoding
    elif embedding=='angle':
        
        def add_layer(angles, n_qubits):
            '''
            Function that applies one layer of the variational anstaz with the given 
            angles on a register of n_qubits with RY, RZ rotations and CNOTs
            
            Parameters
            ----------
            angles : np.array with the rotation angles
            n_qubits : number of qubits of the circuit
        
            Returns
            -------
            None.
        
            '''
                
            for i in range(n_qubits):
                qml.RY(angles[i], wires=i)
            
            for i in range(n_qubits):
                qml.RZ(angles[i+n_qubits], wires=i)
            
            for i in range(n_qubits-1):
                    qml.CNOT(wires=[i,i+1])
        
        n_qubits=4
        
        dev = qml.device('default.qubit', wires=n_qubits)
    
        @qml.qnode(dev)
        def circuit():
            #angle embedding is performed with RX rotations
            qml.AngleEmbedding(feat, wires=range(n_qubits), rotation=rot)
            
            k=0
            for i in range(n_layers):
                add_layer(angles[k:k+2*n_qubits], n_qubits)
                k+=2*n_qubits
            
            return qml.probs(wires = range(n_qubits))
    
    
    probs = circuit()
    
    #we consider as output of the function the sum of the probabilities of the 
    #2nd half of the computational base states
    return sum(probs[int(len(probs)/2):]) 

def cross_entropy(P,Y):
    '''
    Parameters
    ----------
    P : np.array with predicted probabilities
    Y : np.array with the correct labels

    Returns
    -------
    value of the cross_entropy

    '''
    
    eps = 1e-3
    P = np.clip(P, eps, 1 - eps)  #this prevents overflows
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean()
    
def evaluate(angles, dataset, labels, n_layers):
    '''
    Function that evaluates the accuracy of the predictions given angles,
    dataset, labels and  the number of layers

    Parameters
    ----------
    angles : np.array with the rotation angles
    dataset : np.array with data features
    labels : np.array with correct labels
    n_layers :  number of layers of the ansatz

    Returns
    -------
    accuracy : value of the accuracy
    
    '''
    
    predictions = []
    
    #constructing the circuit for every element of the dataset
    for elem in dataset:
        prob = construct_circuit(elem, angles, n_layers) 
        
        if prob > 0.5: # the threshold value for the classification is 0.5
            predictions.append(1)
        else:
            predictions.append(0)
    
    predictions = np.array(predictions)
    
    accuracy = (1-(abs(predictions-labels)).mean()) * 100
   
    return accuracy

def train_classifier(dataset, labels, angles, n_layers, test=None, embedding='amplitude', rot='X'):
    '''
    Function that trains the variational ansatz to act as a classifier given
    the training set, the correct training labels, the initial rotation angles and
    the number of layers of the ansatz. In addition one can also evaluate the
    model perforances during the training by passing test set and labels.

    Parameters
    ----------
    dataset : np.array with training data features
    labels : np.array with correct training labels
    angles : np.array with the initial rotation angles
    n_layers : number of layers of the ansatz
    test : tuple, optional
        First element is the test set and the second are the correct test labels. 
        The default is None.
    embedding : type of embedding of the features passed as a string, optional.
             The default is 'amplitude'. The others are 'angle'.
    rot: String specifying the kind of rotations in the embedding. 
        Only required if 'angle' embedding is selected. The default is 'X'. 
        The others are 'Y', 'Z'. 

    Returns
    -------
    np.array of the optimal rotation angles, list with the trend of train accuracy,
    list with the trend of test accuracy

    '''
    
    def cost(angles):
        
        predictions = []
        pred_probs = []
        
        for elem in dataset:#constructing the circuit for every element of the dataset
            prob = construct_circuit(elem, angles, n_layers, embedding=embedding, rot = rot)
            #print(probs)
            pred_probs.append(prob)
            
            if prob > 0.5: # the threshold value for the classification is 0.5
                predictions.append(1)
            else:
                predictions.append(0)
        
        predictions = np.array(predictions)
        pred_probs = np.array(pred_probs, requires_grad=True)
        
        #calculating the cross entropy from the predicted probabilities
        cross = cross_entropy(pred_probs,labels)
        
        #calculating the accuracy from the predicted labels
        accuracy = (1-(abs(predictions-labels)).mean()) * 100
        accs.append(accuracy)
        print('TRAIN ACCURACY: {}%'.format(round(float(accuracy), 2)))
        
        return cross
    
    if embedding=='angle':
        # rescaling the features in the (0 , 2*np.pi] interval
        max_feat = np.amax(dataset, axis=0)
        dataset = (dataset/max_feat) * 2*np.pi       
    
    accs = []
    test_accs = []
    
    #set the optimizer and number of epochs
    opt = qml.optimize.AdagradOptimizer(.25)
    epochs = 30
    
    for epoch in range(epochs):
        
        print('----------------------')
        angles, value = opt.step_and_cost(cost, angles)#one optimization step
        #angles = np.clip(opt.step(error, angles), -2 * np.pi, 2 * np.pi)
        print(epoch, ':','COST: {}\n'.format(round(float(value), 4)))
        
        if test != None:#if a test set is passed evaluate the model performances

            test_acc = evaluate(angles, test[0], test[1], n_layers)
            test_accs.append(test_acc)
            print('TEST ACCURACY: {}%\n'.format(round(float(test_acc), 2)))
            
    return angles, accs, test_accs



def av_trend(train_csv, test_csv, embedding='amplitude', rot = 'X', n_layers=2, plot=True):
    '''
    Function that calculates the average trend of train and test accuracy. 
    A plot of the trend is shown if variable plot=True.

    Parameters
    ----------
    train_csv : .csv file with train data and labels
    test_csv : .csv file with test data and labels
    embedding : type of embedding of the features passed as a string, optional.
                The default is 'amplitude'. The others are 'angle'.
    rot: String specifying the kind of rotations in the embedding. 
        Only required if 'angle' embedding is selected. The default is 'X'. 
        The others are 'Y', 'Z'. 
    n_layers : number of layers of the ansatz. The default is 2.
    plot : bool, optional
           If True shows the trend plot. The default is True.

    Returns
    -------
    None.

    '''
    
    #reading the train and test files and creating the datasets
    train_set, train_labels = ParseFile(train_csv)
    test_set, test_labels = ParseFile(test_csv)
    
    if embedding == 'amplitude':
        n_qubits = 2
        rotations_layers = 1
        print('AMPLITUDE ENCODING\n')
        
    elif embedding == 'angle':
        n_qubits = 4
        rotations_layers = 2
        print('ANGLE {} ENCODING\n'.format(rot))
    
    #setting random seed to have reproducibility of the results
    np.random.seed(111)
    
    tot_train_accs = []
    tot_test_accs = []
    
    time_list = []
    for step in range(10):
        
        print('START ROUND {} with {} layers\n'.format(step+1, n_layers))
        
        start_time = t.time()
        
        #we initialize the angles at random 
        #then we will analyse the average trend
        #if you want to start with all zeros comment the following line 
        #and uncomment the next one
        angles = 2*np.pi * np.random.rand(rotations_layers*n_layers*n_qubits, requires_grad=True)
        # angles = np.zeros(n_layers*3*n_qubits, requires_grad=True) # if uncommented 
                                                                    # you can change 
                                                                    # the step range to 1
        
        opt_angles, accs, test_accs = train_classifier(train_set, train_labels, angles, n_layers, 
                                      test = (test_set, test_labels), embedding=embedding, rot=rot)
        
        delta_t = t.time() - start_time #measuring the time of the training
        time_list.append(round(delta_t, 1))
        
        tot_train_accs.append(accs)
        tot_test_accs.append(test_accs)
        
        
        #test_acc = evaluate(opt_angles, test_set, test_labels, n_layers)
        #print('\nTEST ACCURACY:', round(float(test_acc), 2), '%\n')
        print('END ROUND {}'.format(step+1))
        print('##################################################################')
    
    av_time = round(sum(time_list)/len(time_list), 2) #average training time
    print('The average training time was: {}'.format(av_time))
    
    ##saving results
    if embedding == 'amplitude':   
        np.save('train_accs_{}_{}layers'.format(embedding,n_layers),tot_train_accs)
        np.save('test_accs_{}_{}layers'.format(embedding,n_layers),tot_test_accs)
        np.save('av_time_{}_{}layers'.format(embedding,n_layers),av_time)
    else:
        np.save('train_accs_{}{}_{}layers'.format(embedding,rot,n_layers),tot_train_accs)
        np.save('test_accs_{}{}_{}layers'.format(embedding,rot,n_layers),tot_test_accs)
        np.save('av_time_{}{}_{}layers'.format(embedding,rot,n_layers),av_time)
    
    #calculating the average trend and the st dev of train e test accuracy
    mean_accs = np.mean(tot_train_accs,axis=0)
    std_accs = np.std(tot_train_accs,axis=0, ddof=1)
    
    mean_test_accs = np.mean(tot_test_accs,axis=0)
    std_test_accs = np.std(tot_test_accs,axis=0, ddof=1)
    
    
    
    if plot:
        #plotting average trend of train e test accuracy
        x = range(len(mean_accs))
        
        fig, ax = plt.subplots()

            
        ax.plot(x, mean_accs, '-', label='Train')
        
        #we plot 
        ax.fill_between(x, mean_accs - std_accs, 
                        list(map(lambda x: x if x<=100 else 100, 
                                 mean_accs + std_accs)), 
                        alpha=0.2)
        
        ax.plot(x, mean_test_accs, '-', label='Test')
        ax.fill_between(x, mean_test_accs - std_test_accs, 
                        list(map(lambda x: x if x<=100 else 100, 
                                 mean_test_accs + std_test_accs)), 
                        alpha=0.1)

        if embedding == 'amplitude':        
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy trend with {} embedding and {} layers'.format(embedding,
                                                                              n_layers))
            
            fig.savefig('mean_train_accs_{}_{}_lay.png'.format(embedding,n_layers))
        else:
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Accuracy trend with {}{} embedding and {} layers'.format(embedding,rot,
                                                                              n_layers))
            
            fig.savefig('mean_train_accs_{}{}_{}_lay.png'.format(embedding,rot,n_layers))
        
    return

if __name__ == "__main__":
    
    # #calculating the average accuracy trend for 1, 2, and 3 layers
    # #for amplitude embedding
    
    max_n_layers = 3
    
    for i in range(max_n_layers):
        
        av_trend('mock_train_set.csv','mock_test_set.csv', embedding='amplitude', 
            n_layers=i+1)
    
    ##for angleX embedding
    for i in range(max_n_layers):
        
        av_trend('mock_train_set.csv','mock_test_set.csv', embedding='angle', 
            rot='X',n_layers=i+1)
    
    ##for angleY embedding
    for i in range(max_n_layers):
        
        av_trend('mock_train_set.csv','mock_test_set.csv', embedding='angle', 
            rot='Y', n_layers=i+1)
    
    ##for angleZ embedding
    for i in range(max_n_layers):
        av_trend('mock_train_set.csv','mock_test_set.csv', embedding='angle', 
            rot='Z', n_layers=i+1)
    
    
    
    
    
    
    
    
    
    
    