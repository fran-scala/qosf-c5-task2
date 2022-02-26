# qosf cohort 5 

# Task 2 : Encoding and Classifier

## Introduction

Hi! This repository contains my submission to *qosf QC Mentorship program*. I dealt with Task 2 that is the following:


- Encoding the following files in a quantum circuit [mock_train_set.csv](https://drive.google.com/file/d/1PIcC1mJ_xi4u1-2gxyoStg2Rg_joSBIB/view?usp=sharing) and [mock_test_set.csv](https://drive.google.com/file/d/1aapYE69pTeNHZ6u-qKAoLfd1HWZVWPlB/view?usp=sharing) in at least two different ways (these could basis, angle, amplitude, kernel or random encoding).
- Design a variational quantum circuit for each of the encodings, uses the column 4 as the target, this is a binary class 0 and 1.
- You must use the data from column0 to column3 for your proposed classifier.
- Consider the ansatz you are going to design as a layer and find out how many layers are
necessary to reach the best performance. Analyze and discuss the results.

Feel free to use existing frameworks (e.g. PennyLane, Qiskit) for creating and training the circuits.

This PennyLane demo can be useful: [Training a quantum circuit with Pytorch](https://pennylane.ai/qml/demos/tutorial_state_preparation.html),

This Quantum Tensorflow tutorial can be useful: [Training a quantum circuit with Tensorflow](https://www.tensorflow.org/quantum/tutorials/mnist).

For the variational circuit, you can try any circuit you want. You can start from one with a layer of RX, RZ and CNOTs.

### Required packages

In order to properly execute everything `Pennylane 0.21.0` and `Matplotlib 3.4.3` are needed.    

### Repository contents

The repository contains `main.py` that performs the main computation and `cfr_plots.py` which gives some plots useful for the analysis of the different training processes. All the data related to the trainings were already saved and are available in the repository to run the `cfr_plots.py` file. In addition, there is also the `Images` folder that contains the all the obtained plots.

## Encodings

### Angle encoding

### IQP encoding


### Amplitude encoding







