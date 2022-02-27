#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 16:19:09 2022

@author: francesco
"""
import pennylane as qml
from pennylane import numpy as np
import time as t
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
##loading amplitude embedding test trends
amp_test = {}

for i in range(3):
    amp_test['test_amp_{}_lay'.format(i+1)] = np.load('test_accs_amplitude_{}layers.npy'.format(i+1))

means_amp = {}

for i in range(3):
    means_amp['mean_test_accs_amp_{}'.format( i+1)] = np.mean(amp_test['test_amp_{}_lay'.format(i+1)], axis=0)


#plotting the trends to have a comparison of the performances
fig, ax = plt.subplots()

x = range(len(means_amp['mean_test_accs_amp_1']))

for i in range(3):
    ax.plot(x, means_amp['mean_test_accs_amp_{}'.format(i+1)], '-', label='{} layers'.format(i+1))

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test accuracy trend with amplitude embedding')
          
# #-----------------------------------------------------------------------------
##loading angle embedding test trends

ang_test = {}
rotations = ['X', 'Y', 'Z']


for rot in rotations:
    for i in range(3):
        ang_test['test_ang{}_{}_lay'.format(rot, i+1)] = np.load('test_accs_angle{}_{}layers.npy'.format(rot,i+1))


means_ang = {}
for rot in rotations:
    for i in range(3):
        means_ang['mean_test_accs_ang{}_{}'.format(rot, i+1)] = np.mean(ang_test['test_ang{}_{}_lay'.format(rot, i+1)], axis=0)


#plotting the trends to have a comparison of the performances
fig, ax = plt.subplots()

x = range(len(means_ang['mean_test_accs_angX_1']))

for rot in rotations:
    for i in range(3):
        ax.plot(x, means_ang['mean_test_accs_ang{}_{}'.format(rot,i+1)], '-', label='{} - {} layers'.format(rot, i+1))

plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Test accuracy trend with angle embedding')

# #-----------------------------------------------------------------------------
##loading average training time for different embedddings 

embedding = ['amplitude', 'angle']

av_time = {}

for emb in embedding:
        if emb == 'amplitude':
            for i in range(3):
                av_time[emb] = av_time.get(emb, []) +[np.load('av_time_{}_{}layers.npy'.format(emb, i+1))]
        else:
            for rot in rotations:
                for i in range(3):
                    
                    av_time['{}{}'.format(emb, rot)] = av_time.get('{}{}'.format(emb, rot), []) +[np.load('av_time_{}{}_{}layers.npy'.format(emb,rot, i+1))]
        

#plotting the average training time for different embedding as a function of 
#the number of layers
x = range(1,4)

fig, ax = plt.subplots()
ax.xaxis.set_major_locator(plt.MultipleLocator(1))

for key, value in av_time.items():
    plt.plot(x, value, '-', label=key)

plt.legend()

plt.xlabel('Layers')
plt.ylabel('Seconds')
plt.title('Average training time')














