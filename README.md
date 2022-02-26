# qosf cohort 5 

<p>
  <a href="" target="_blank"><img src="https://github.com/fran-scala/qosf-c5-task2/blob/7be030043a89a70ba63dfeae95a818950bac6975/qosf_logo.png?raw=True" width="30%"/> </a>
</p>

# Task 2 : Encoding and Classifier

## Introduction

Hi! This repository contains my submission to ***qosf QC Mentorship program***. I dealt with Task 2 that is the following:


- Encoding the following files in a quantum circuit [mock_train_set.csv](https://drive.google.com/file/d/1PIcC1mJ_xi4u1-2gxyoStg2Rg_joSBIB/view?usp=sharing) and [mock_test_set.csv](https://drive.google.com/file/d/1aapYE69pTeNHZ6u-qKAoLfd1HWZVWPlB/view?usp=sharing) in at least two different ways (these could basis, angle, amplitude, kernel or random encoding).
- Design a variational quantum circuit for each of the encodings, uses the column 4 as the target, this is a binary class 0 and 1.
- You must use the data from column 0 to column 3 for your proposed classifier.
- Consider the ansatz you are going to design as a layer and find out how many layers are
necessary to reach the best performance. Analyze and discuss the results.

Feel free to use existing frameworks (e.g. PennyLane, Qiskit) for creating and training the circuits.

This PennyLane demo can be useful: [Training a quantum circuit with Pytorch](https://pennylane.ai/qml/demos/tutorial_state_preparation.html),

This Quantum Tensorflow tutorial can be useful: [Training a quantum circuit with Tensorflow](https://www.tensorflow.org/quantum/tutorials/mnist).

For the variational circuit, you can try any circuit you want. You can start from one with a layer of RX, RZ and CNOTs.

### Required packages

In order to properly execute everything `Pennylane 0.21.0` and `Matplotlib 3.4.3` are needed.    

### Repository contents

The repository contains the data files `mock_train_set.csv` and `mock_test_set.csv`, the Python scripts `main.py` that performs the main computation and `cfr_plots.py` which gives some plots useful for comparing different training processes. Things works properly when one first run the `main.py` file and then the `cfr_plots.py` file. Anyway, all the info collected during training and testing were already saved and are made available in the repository (`*.npy`), so one can directly run the `cfr_plots.py` file. In addition, there is also the `Images` folder containing the all the obtained plots. The complete result analysis is reported in `qosf_c5_task2.pdf` while the full cohort 5 description of the tasks is provided in the `Cohort_5_Screening_Tasks.pdf`.

### Summary of the approach

After having read the data files, we encode the data features with different encoding techniques and we apply a variational circuit to classify the data items. Since, for sake of generality, we initialize the parameters of the classifier at random we will repeat the training procedure multiple times and in the end we will look at the average trend. We also repeat the same procedure for different numbers of ansatz layers in the variational circuit.

## Encodings

### Angle encoding
[Angle embedding](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.AngleEmbedding.html) encodes *N* features into the rotation angles of *n* qubits, where *N ≤ n*. In general, if there are fewer features than rotations, the remaining rotation gates are not applied. In the case considered, we use for 4 qubits to encode the four features. We renormalize the features so as to have values in *(0 , 2 pi]*.

The rotations can be chosen as either *RX, RY* or *RZ* gates.


### Amplitude encoding

[Amplitude embedding](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.AmplitudeEmbedding.html) encodes *2^n* features into the amplitude vector of *n* qubits. To represent a valid quantum state vector, the L2-norm of features must be one.  In the case considered, we only needed 2 qubits to encode the four features. This will result in a substantial speedup in terms of training time.

## Principal functions in `main.py`

In order to read the train and test data, we at first define the `ParseFile` which returns the dataset with data features as a np.array and the correct labels as a np.array. Then we have `construct_circuit` that is a function encoding the features with the specified embedding and then applies the variational ansatz. For what concerns the variational ansatz we used two different ansatzes depending on the encoding. 

For the Amplitude Embedding we used as a classifier a variational circuit only made of *RY* rotations and one *CNOT* gate.

![amp_emb+ansatz](https://github.com/fran-scala/qosf-c5-task2/blob/f243e75a9cb24f5e0430e7df7025ac889eea1873/Images/circuit_amp_emb.png?raw=True)

Since in the case of Angle Embedding the previous ansatz seemed to struggle in classyfing correctly, we tried to increase its complexity by adding to the *RY* rotations also a layer of *RZ* rotations. Then three *CNOT* gates are applied linearly. It should be noticed that by combining *RY* and *RZ* the whole Bloch sphere will be spanned for each qubit.

![ang_emb+ansatz](https://github.com/fran-scala/qosf-c5-task2/blob/f243e75a9cb24f5e0430e7df7025ac889eea1873/Images/circuit_ang_emb.png?raw=True)

We considered as output of the function the sum of the probabilities of the 2nd half of the computational base states.

To train our classifier we defined a dedicated function `train_classifier` that progressively updates the variational ansatz parameters to act as a classifier given the training set, the correct training labels, the initial rotation angles and the number of layers of the ansatz. In addition, one can also evaluate the model performances during the training by passing test set and labels. As a cost function we use the cross entropy calculated by giving in input the array containing all the output of the computation and the array of correct labels. During the training, we also display the the accuracy of the classification performed with the parameters at each step. One data item is considered to be of class 1 if the output of the `construct_circuit` function is greater than 0.5, otherwise it is considered as belonging at class 0. As an optimizer we chose [Adagrad](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.AdagradOptimizer.html), that is a gradient-descent optimizer with past-gradient-dependent learning rate in each dimension, meaning that it adjusts the learning rate for each parameter *x_i* in the parameter vector *x* based on past gradients. We set the the stepsize at *0.25* and we repeat the training process for *30* epochs.

Last but not least, we implemented the function that calculates the average trend of train and test accuracy, plotting them if `plot=True`. The function reads the train and test `.csv` files and after having randomly initialized the parameters of the classifier it calls the `train_classifier` function. Here we set a random seed in order to have reproducibility of the data. This is repeated *10* times. In addition, here we keep track of the training time in order to compare the time requirements of each approach.




