import numpy as np
from numpy.fft import fft, fftfreq, ifft
from HHGtoolkit.utilities import Data, Dataset
import HHGtoolkit.utilities as util
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import h5py
from scipy.signal.windows import *
from math import erf
import cmath
import random

### Generate random indeces according to the exponential distribution
def generate_random_integers(N, lambda_ = 5, max_index = 1000):
    # Initialize an empty list to store the random integers
    random_integers = []
    # Loop N times
    for i in range(N):
        # Generate a random number between 0 and 1 using the exponential cumulative distribution function
        # The parameter lambda is set to 0.001 for demonstration purposes
        # You can change it to any positive value you want
        random_number = random.expovariate(lambda_)
        # Multiply the random number by 1000 and round it to the nearest integer
        random_integer = round(random_number*max_index)
        while (random_integer > max_index-1):
            random_number = random.expovariate(lambda_)
            random_integer = round(random_number*1000)
        # Append the random integer to the list
        random_integers.append(random_integer)
    # Return the list of random integers
    return random_integers

### Complex n_th_root
def n_th_root(arr, n):
    arr = [np.sign(x)*np.abs(x)**n for x in arr]
    arr = [cmath.rect(np.abs(x), cmath.phase(x)).real for x in arr]
    return np.array(arr)

### Real and imaginary part separation
def fft_real_imag_separation(arr, t, n = 1):
    N = len(arr)
    dt = t[1]-t[0]
    T = t[-1]
    freqs = fftfreq(len(arr), dt)*2*np.pi
    FF = fft(arr)
    FF *= np.exp(1j*freqs*T/2)
    real_ = FF.real[0:N//2+1]
    imag_ = (FF.imag[0:N//2+1])
    if n == 1:
        return real_, imag_
    real_ = n_th_root(real_, 1/n)
    imag_ = n_th_root(imag_, 1/n)
    return real_, imag_

### Real and imaginary part reconstruction
def fft_real_imag_reconstruction(real, imag, t, n = 1, shift = 0.):
    dt = t[1]-t[0]
    T = t[-1]
    N = 2*len(real)-1
    res = 1j*np.zeros((N))
    FF = real**n + 1j*(imag)**n
    FF_conj = np.flip(np.conj(FF)[1:N//2+1])
    res[0:N//2+1] = FF
    res[N//2+1:N] = FF_conj
    freqs = fftfreq(len(res), dt)*2*np.pi
    res = res*np.exp(-1j*freqs*T/2)*np.exp(1j*freqs*shift)
    return ifft(res).real

### Separation of real and imaginary parts and frequency clip
def preprocess(signal, size, t, n = 1):
    N_original = len(signal)
    T = t[-1]
    FT = fft(signal)
    ### Find corresponding range given the input size
    range_ = np.r_[0:((size)//2+1), (-(size)//2+1):0]
    ### Select the signal according to range
    t_f = np.linspace(0, T, size)
    dt = t_f[1]-t_f[0]
    freqs = fftfreq(size, dt)*2*np.pi
    FT = FT[range_]*np.exp(1j*freqs*T/2*1)
    ### Normalize the signal correspondingly
    FT = size/N_original*FT
    ### Separate to real and imaginary part
    real_ = FT.real[0:size//2+1]
    imag_ = FT.imag[0:size//2+1]
    if n == 1:
        return real_, imag_
    real_ = n_th_root(real_, 1/n)
    imag_ = n_th_root(imag_, 1/n)
    return real_, imag_

### Load E and grad V from hdf5 datasets
def load_fields_from_h5(dir):
    arr = os.listdir(dir)
    file_paths = [dir + x for x in arr if x.endswith('.h5')]
    util.sort_nicely(file_paths)
    dset = list()
    for file in file_paths:
        data = h5py.File(file, "r")
        dset.append([data["outputs/Efield"][:,0], data["outputs/grad_pot"][:,0]])
        data.close()
    data = h5py.File(file, "r")
    t = data["outputs/tgrid"][:]
    data.close()
    return np.array(dset), t

### Reconstruct the original pulse from the real and imaginary parts
def reconstruct(real, imag, t, n = 1, shift = 0.):
    dt = t[1]-t[0]
    T = t[-1]
    #T = 0
    N = 2*len(real)-1
    res = 1j*np.zeros((N))
    FF = real**n + 1j*(imag)**n
    FF = real + 1j*imag
    FF_conj = np.flip(np.conj(FF)[1:N//2+1])
    res[0:N//2+1] = FF
    res[N//2+1:N] = FF_conj
    freqs = fftfreq(len(res), dt)*2*np.pi
    res = res*np.exp(-1j*freqs*T/2*1)*np.exp(-1j*freqs*shift)
    return ifft(res).real

def filter(arr, t, slope_coef=14.8, shift_coef=4.9, slope2_coef = 20.5, shift2_coef = 6.8):
    t_max = t[-1]
    M = len(t)
    shift = t_max/shift_coef
    slope = t_max/slope_coef
    shift2 = t_max/shift2_coef
    slope2 = t_max/slope2_coef
    arr[:M//2] = arr[:M//2] * (1+np.array(list(map(erf, (t[:M//2]-shift2)/(slope2)))))/2
    arr[M//2:] = arr[M//2:] * (1-np.array(list(map(erf, (t[M//2:]-t_max+shift)/(slope)))))/2
    return arr

def min_max_normalize(x_train, x_test):
    # Assume x_train is a tensor of shape (batch_size_train, N, 2)
    x_train_min = tf.reduce_min(x_train) # scalar
    x_train_max = tf.reduce_max(x_train) # scalar
    x_train = 2 * (x_train - x_train_min) / (x_train_max - x_train_min) - 1 # shape (batch_size_train, N, 2)

    # Assume x_val is a tensor of shape (batch_size_val, N, 2)
    x_test = 2 * (x_test - x_train_min) / (x_train_max - x_train_min) - 1 # shape (batch_size_val, N, 2)
    return x_train, x_test

class TDSE_CNN(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, dropout = 0.5, activation = "tanh", conv_activation = "tanh"):
        """TDSE NN constructor"""
        super().__init__()
        
        self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.nonlinear1 = tf.keras.layers.Dense(128, activation=activation)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.nonlinear2 = tf.keras.layers.Dense(64, activation=activation)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.nonlinear3 = tf.keras.layers.Dense(32, activation=activation)
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        self.nonlinear4 = tf.keras.layers.Dense(64, activation=activation)
        self.dropout4 = tf.keras.layers.Dropout(dropout)
        self.nonlinear5 = tf.keras.layers.Dense(128, activation=activation)
        self.dropout5 = tf.keras.layers.Dropout(dropout)
        
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        self.output_layer = tf.keras.layers.Dense(self.output_size, activation="linear")
    
        ### Flattening layer
        self.flat = tf.keras.layers.Flatten() 
        ### Covolutional layers
        self.conv1 = tf.keras.layers.Conv1D(8, 3, activation=conv_activation)
        self.conv2 = tf.keras.layers.Conv1D(16, 3, activation=conv_activation)
        self.conv3 = tf.keras.layers.Conv1D(32, 3, activation=conv_activation)
        self.conv4 = tf.keras.layers.Conv1D(32, 3, activation=conv_activation)
        
        ### Pooling layers
        self.max_pooling1 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling2 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling3 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling4 = tf.keras.layers.MaxPooling1D(3)
        
        #self.max_pooling1 = tf.keras.layers.AvgPool1D(3)
        #self.max_pooling2 = tf.keras.layers.AvgPool1D(3)
        #self.max_pooling3 = tf.keras.layers.AvgPool1D(3)
        #self.max_pooling4 = tf.keras.layers.AvgPool1D(3)
        
        self.batch_normalization = tf.keras.layers.BatchNormalization()     
        
        ### Adding layer
        self.add = tf.keras.layers.Add()  
        ### Tanh activation layer
        self.tanh = tf.keras.layers.Activation("tanh")

        ### Initial noise layer
        #self.noise1 = tf.keras.layers.GaussianNoise(0.3)
        self.noise1 = tf.keras.layers.GaussianNoise(0.1)
        

    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        
        ### Convolution section
        #x = self.max_pooling4(x)
        x = self.conv1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        x = self.max_pooling2(x)
        x = self.conv3(x)
        x = self.max_pooling3(x)
        #x = self.conv4(x)
        
        ### Flattening layer
        x = self.flat(x)
        
        ### Normalize batch
        x = self.batch_normalization(x)

        ### MLP section
        x = self.nonlinear1(x)
        ### Skipped connection
        shortcut = x
        ### Introduce some noise
        x = self.noise1(x)
        x = self.nonlinear2(x)
        x = self.dropout1(x)
        x = self.nonlinear3(x)
        x = self.dropout2(x)
        x = self.nonlinear4(x)
        x = self.dropout3(x)
        x = self.nonlinear5(x)
        x = self.dropout4(x)
        ### Reconnect with the skipped connection
        x = self.add([x, shortcut])
        ### Apply tanh
        x = self.tanh(x)
        x = self.dropout5(x)

        ### Output layer
        x = self.output_layer(x)
        
        return x
        