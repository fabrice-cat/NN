"""@package NeuralNetwork
Implementation of a neural network (NN) for source term computation for the
coupled model.

"""

import tensorflow as tf
import numpy as np
from numpy.fft import fft, ifft, fftfreq

#"""For more info why this line is useful see:
#https://oliver-k-ernst.medium.com/a-cheat-sheet-for-custom-tensorflow-layers-and-models-aa465df2bc8b
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/register_keras_serializable
#https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
#"""
#@tf.keras.utils.register_keras_serializable(package="NeuralNetwork")
class TDSE_NN(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, size):
        """TDSE NN constructor"""
        super().__init__()
        self.size = size
        ### First input layer
        #self.input_layer = tf.keras.layers.InputLayer()
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        ### Simple linear output dense layer with linear activation function 
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        ### f(y) = y, y = W.x + b
        self.output_layer = tf.keras.layers.Dense(size)


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        #x = self.input_layer(inputs)
        #x = self.preprocess(x)
        x = self.nonlinear1(inputs)
        x = self.dropout1(x)
        x = self.nonlinear2(x)
        x = self.dropout2(x)
        x = self.nonlinear3(x)
        x = self.dropout3(x)
        return self.output_layer(x)
    

    #"""Function good for a custom layer - contains configuration of a layer
    #more info: https://keras.io/guides/serialization_and_saving/#custom-objects and
    #https://oliver-k-ernst.medium.com/a-cheat-sheet-for-custom-tensorflow-layers-and-models-aa465df2bc8b
    #"""
    #def get_config(self):
    #    config = super(TDSE_NN, self).get_config()
    #    config.update({
    #        "size": self.size,            
    #        "nonlinear1": self.nonlinear1,
    #        "nonlinear2": self.nonlinear2,
    #        "dropout1": self.dropout1,
    #        "dropout2": self.dropout2,
    #        "output_layer": self.output_layer
    #        })
    #    return config
        
    #@classmethod
    #def from_config(cls, config):
    #    return cls(**config)


#@tf.keras.utils.register_keras_serializable(package="NeuralNetwork")
class Preprocessing(tf.keras.layers.Layer):
    """Data preprocessing layer

    This layer does the initial data preprocessing. It does spectral filtering
    of the input to a fixed sized output of size N. 
    """
    def __init__(self, size):
        super().__init__()
        self.trainable = False
        self.units = size
        #self.input_shape = (1000,)

    #def get_config(self):
    #    config = super(Preprocessing, self).get_config()
    #    config.update({
    #        "trainable": self.trainable,
    #        "units": self.units
    #        })
    #    return config

    #@classmethod
    #def from_config(cls, config):
    #    return cls(**config)

    #def call(self, inputs):
    #    return spectral_filter(inputs, self.units)

    #def build(self, input_shape):
    #    self.w = self.add_weight(
    #        shape=(input_shape[-1], self.units),
    #        initializer="random_normal",
    #        trainable=True,
    #    )
    #    self.b = self.add_weight(
    #        shape=(self.units,), initializer="random_normal", trainable=True
    #    )

     #   return super().build(input_shape)

def spectral_filter(signal, size):
    N_original = len(signal)
    ### FFT of the signal
    FT = fft(signal)
    ### Find corresponding range given the input size
    range_ = np.r_[0:((size)//2+1), (-(size)//2+1):0]
    #range_ = (omegas >= -harmonic_cutoff*omega_0) & (omegas <= harmonic_cutoff*omega_0)
    ### Select the signal according to range
    signal_filtered = ifft(FT[range_]).real
    ### Normalize the signal correspondingly
    signal_filtered = size/N_original*signal_filtered
    return signal_filtered

### Experimental
class TDSE_NN_2(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, N = 257, activation = tf.nn.tanh):
        """TDSE NN constructor"""
        super().__init__()
        #self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(0.6)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(N, activation=activation, use_bias=False)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(N, activation=activation)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(N, activation=activation)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(N, activation=activation)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(0.5)
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        self.output_layer = tf.keras.layers.Dense(self.input_shape_[-1], use_bias=False)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten()


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        #x = self.preprocess(x)
        #x = self.flatten(x)
        x = self.dropout0(x)
        x = self.nonlinear1(x)
        x = self.dropout1(x)
        x = self.nonlinear2(x)
        x = self.dropout2(x)
        x = self.nonlinear3(x)
        x = self.dropout3(x)
        #x = self.nonlinear4(x)
        #x = self.dropout4(x)
        x = self.output_layer(x)
        #return self.flatten(x)
        return x
        #return [x[0:self.input_shape_[1]], x[self.input_shape_[1]+1:]]
    

class TDSE_NN_3(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, N = 256, activation = tf.nn.tanh, bias = False):
        """TDSE NN constructor"""
        super().__init__()
        #self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(0.8)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(0.5)
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        self.output_layer = tf.keras.layers.Dense(self.input_shape_[-1], use_bias=bias)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten()


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        #x = self.preprocess(x)
        #x = self.flatten(x)
        x = self.dropout0(x)
        x = self.nonlinear1(x)
        x = self.dropout1(x)
        x = self.nonlinear2(x)
        #x = self.dropout2(x)
        #x = self.nonlinear3(x)
        #x = self.dropout3(x)
        #x = self.nonlinear4(x)
        x = self.dropout4(x)
        x = self.output_layer(x)
        #return self.flatten(x)
        return x
        #return [x[0:self.input_shape_[1]], x[self.input_shape_[1]+1:]]
    

class TDSE_NN_4(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, N = 256, activation = tf.nn.tanh, bias = False, dropout = 0.5, dropout_input = 0.6):
        """TDSE NN constructor"""
        super().__init__()
        self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(dropout_input)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(dropout)
        ### Fifth nonlinear dense layer
        self.nonlinear5 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout5 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear6 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout6 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear7 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout7 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear8 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout8 = tf.keras.layers.Dropout(dropout)
        self.nonlinear9 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout9 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear10 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout10 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear11 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout11 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear12 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout12 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear13 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout13 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear14 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        #self.dropout14 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear15 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        #self.dropout15 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear16 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        #self.dropout16 = tf.keras.layers.Dropout(dropout)
        #self.nonlinear17 = tf.keras.layers.Dense(N, activation=activation, use_bias=bias)
        #self.dropout17 = tf.keras.layers.Dropout(dropout)
        #self.linear = tf.keras.layers.Dense(100)
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        
        self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias, activation=tf.keras.activations.tanh)
        #self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten(data_format = "channels_first")
        self.flatten_inp = tf.keras.layers.Flatten()
        ### Covolutional layer
        self.convolution1 = tf.keras.layers.Conv1D(8, 5)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))
        self.convolution2 = tf.keras.layers.Conv1D(16, 5)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))
        self.convolution3 = tf.keras.layers.Conv1D(32, 5)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))
        #self.avg_pooling1 = tf.keras.layers.MaxPooling1D(5)
        #self.avg_pooling1 = tf.keras.layers.AveragePooling1D(10)
        self.avg_pooling2 = tf.keras.layers.AveragePooling1D(5)
        self.avg_pooling3 = tf.keras.layers.AveragePooling1D(5)
        self.normalization = tf.keras.layers.Normalization()


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        #x = tf.transpose(x)
        #x = self.flatten_inp(x)
        #x = self.preprocess(x)
        #x_reshaped = tf.expand_dims(x, axis=2)
        #x_reshaped = tf.expand_dims(x, axis=2)
        
        #x = self.avg_pooling3(x_reshaped)
        #x = self.avg_pooling1(x)
        #x = self.convolution1(x)
        #x = self.avg_pooling2(x)
        #x = self.convolution2(x)
        #x = self.avg_pooling3(x)
        #x = self.convolution3(x)
       
        x = self.flatten(x)
        #x = self.normalization(x)
        x = self.dropout0(x)
        x = self.nonlinear1(x)
        x = self.dropout1(x)
        x = self.nonlinear2(x)
        x = self.dropout2(x)
        x = self.nonlinear3(x)
        x = self.dropout3(x)
        x = self.nonlinear4(x)
        x = self.dropout4(x)
        x = self.nonlinear5(x)
        x = self.dropout5(x)
        x = self.nonlinear6(x)
        x = self.dropout6(x)
        #x = self.nonlinear7(x)
        #x = self.dropout7(x)
        #x = self.nonlinear8(x)
        #x = self.dropout8(x)
        #x = self.nonlinear9(x)
        #x = self.dropout9(x)
        #x = self.nonlinear10(x)
        #x = self.dropout10(x)
        #x = self.nonlinear11(x)
        #x = self.dropout11(x)
        #x = self.nonlinear12(x)
        #x = self.dropout12(x)
        #x = self.nonlinear13(x)
        #x = self.dropout13(x)
        #x = self.nonlinear14(x)
        #x = self.dropout14(x)
        #x = self.nonlinear15(x)
        #x = self.dropout15(x)
        #x = self.nonlinear16(x)
        #x = self.dropout16(x)
        #x = self.nonlinear17(x)
        #x = self.dropout17(x)
        x = self.output_layer(x)
        #return self.flatten(x)
        return x
        #return x
    

class TDSE_NN_5(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, N = 256, activation = tf.nn.tanh, bias = False, dropout = 0.5, dropout_input = 0.6):
        """TDSE NN constructor"""
        super().__init__()
        self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(dropout_input)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(1, activation=activation, use_bias=bias)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(4, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(16, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(dropout)
        ### Fifth nonlinear dense layer
        self.nonlinear5 = tf.keras.layers.Dense(256, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout5 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        #self.nonlinear6 = tf.keras.layers.Dense(1, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout6 = tf.keras.layers.Dropout(dropout)
        
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        
        self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias, activation="linear")
        #self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten(data_format = "channels_first")        


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        x = self.flatten(x)
        #x = tf.transpose(x)
        #x = self.flatten_inp(x)
        #x = self.preprocess(x)
        #x_reshaped = tf.expand_dims(x, axis=2)
        #x_reshaped = tf.expand_dims(x, axis=2)
        
        #x = self.avg_pooling3(x_reshaped)
        #x = self.avg_pooling1(x)
        #x = self.convolution1(x)
        #x = self.avg_pooling2(x)
        #x = self.convolution2(x)
        #x = self.avg_pooling3(x)
        #x = self.convolution3(x)
       
        
        x = self.dropout0(x)
        x = self.nonlinear1(x)
        x = self.dropout1(x)
        x = self.nonlinear2(x)
        x = self.dropout2(x)
        x = self.nonlinear3(x)
        x = self.dropout3(x)
        x = self.nonlinear4(x)
        x = self.dropout4(x)
        x = self.nonlinear5(x)
        #x = self.dropout5(x)
        #x = self.nonlinear6(x)
        x = self.flatten(x)
        x = self.dropout6(x)
        #x = self.nonlinear7(x)
        #x = self.dropout7(x)
        #x = self.nonlinear8(x)
        #x = self.dropout8(x)
        #x = self.nonlinear9(x)
        #x = self.dropout9(x)
        #x = self.nonlinear10(x)
        #x = self.dropout10(x)
        #x = self.nonlinear11(x)
        #x = self.dropout11(x)
        #x = self.nonlinear12(x)
        #x = self.dropout12(x)
        #x = self.nonlinear13(x)
        #x = self.dropout13(x)
        #x = self.nonlinear14(x)
        #x = self.dropout14(x)
        #x = self.nonlinear15(x)
        #x = self.dropout15(x)
        #x = self.nonlinear16(x)
        #x = self.dropout16(x)
        #x = self.nonlinear17(x)
        #x = self.dropout17(x)
        x = self.output_layer(x)
        #return self.flatten(x)
        return x
        #return x
    

class TDSE_NN_Conv(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, bias = False, dropout = 0.5, dropout_input = 0.6, activation = "tanh", conv_activation = "tanh"):
        """TDSE NN constructor"""
        super().__init__()
        #activation = tf.nn.relu
        self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(dropout_input)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(128, activation=activation, use_bias=bias)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(32, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(dropout)
        ### Fifth nonlinear dense layer
        self.nonlinear5 = tf.keras.layers.Dense(128, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout5 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        #self.nonlinear6 = tf.keras.layers.Dense(1, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout6 = tf.keras.layers.Dropout(dropout)
        self.nonlinear6 = tf.keras.layers.Dense(256, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout6 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear7 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout7 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear8 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout8 = tf.keras.layers.Dropout(dropout)
        self.nonlinear9 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout9 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear10 = tf.keras.layers.Dense(32, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout10 = tf.keras.layers.Dropout(dropout)

        self.input_dense = tf.keras.layers.Dense(self.output_size, activation=activation)
        
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        
        self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias, activation="linear")
        #self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten(data_format = "channels_first") 
        self.flat = tf.keras.layers.Flatten() 
        ### Covolutional layer
        self.convolution1 = tf.keras.layers.Conv1D(8, 5)#, activation=activation)#, input_shape = (None, *self.input_shape_))
        self.convolution2 = tf.keras.layers.Conv1D(16, 5)#, activation=activation)#, input_shape = (None, *self.input_shape_))
        self.convolution3 = tf.keras.layers.Conv1D(32, 5)#, activation=activation)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))
        self.convolution4 = tf.keras.layers.Conv1D(64, 3)#, activation=activation)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))

        self.conv0 = tf.keras.layers.Conv1D(4, 3, activation=conv_activation)
        self.conv1 = tf.keras.layers.Conv1D(8, 3, activation=conv_activation)
        self.conv2 = tf.keras.layers.Conv1D(16, 3, activation=conv_activation)
        self.conv3 = tf.keras.layers.Conv1D(32, 3, activation=conv_activation)
        #self.max_pooling1 = tf.keras.layers.MaxPooling1D(5)
        self.max_pooling0 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling1 = tf.keras.layers.MaxPooling1D(3)
        #self.max_pooling2 = tf.keras.layers.MaxPooling1D(5)
        self.max_pooling2 = tf.keras.layers.MaxPooling1D(3)
        #self.max_pooling3 = tf.keras.layers.MaxPooling1D(5)
        self.max_pooling3 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling4 = tf.keras.layers.MaxPooling1D(3)
        self.avg_pooling1 = tf.keras.layers.AveragePooling1D(3)
        self.avg_pooling2 = tf.keras.layers.AveragePooling1D(3)
        self.avg_pooling3 = tf.keras.layers.AveragePooling1D(3)  
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization4 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization5 = tf.keras.layers.BatchNormalization()     
        #self.batch_normalization5 = tf.keras.layers.BatchNormalization()     
        #self.batch_normalization5 = tf.keras.layers.BatchNormalization()     
        #self.batch_normalization5 = tf.keras.layers.BatchNormalization()   
        self.add = tf.keras.layers.Add()  
        self.tanh = tf.keras.layers.Activation("tanh")
        self.noise0 = tf.keras.layers.GaussianNoise(0.3)
        #self.noise1 = tf.keras.layers.GaussianNoise(0.1)
        self.noise1 = tf.keras.layers.GaussianNoise(0.3)
        self.noise2 = tf.keras.layers.GaussianNoise(0.1)
        self.noise3 = tf.keras.layers.GaussianNoise(0.1)
        self.noise4 = tf.keras.layers.GaussianNoise(0.1)
        self.noise5 = tf.keras.layers.GaussianNoise(0.1)
        self.noise6 = tf.keras.layers.GaussianNoise(0.1)
        #self.noise7 = tf.keras.layers.GaussianNoise(0.1)
        self.noise7 = tf.keras.layers.GaussianNoise(0.3)
        self.dense_add = tf.keras.layers.Dense(128)


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        #x = tf.transpose(x)
        #x = self.flatten_inp(x)
        #x = self.preprocess(x)
        #x_reshaped = tf.expand_dims(x, axis=2)
        
        #x = self.avg_pooling3(x_reshaped)
        #x = self.max_pooling1(x)
        #x = self.noise0(x)
        #x = self.conv0(x)
        #x = self.max_pooling0(x)
        #shortcut = self.flatten(x)
        #x = tf.expand_dims(x, axis=2)
        #x = self.convolution1(x)

        x = self.conv1(x)
        #x = self.batch_normalization2(x)
        #x = self.avg_pooling1(x)
        x = self.max_pooling1(x)
        x = self.conv2(x)
        #x = self.batch_normalization3(x)
        #x = self.convolution2(x)
        #x = self.avg_pooling2(x)
        x = self.max_pooling2(x)
        x = self.conv3(x)
        #x = self.batch_normalization4(x)
        #x = self.convolution3(x)
        #x = self.avg_pooling3(x)
        x = self.max_pooling3(x)
        #x = self.flatten(x)
        x = self.flat(x)
        #x = self.input_dense(x)
        #shortcut = x
        
        #x = self.dropout0(x)
        x = self.batch_normalization1(x)
        x = self.nonlinear1(x)
        shortcut = x
        x = self.noise1(x)
        x = self.nonlinear2(x)
        x = self.dropout1(x)
        #x = self.batch_normalization2(x)
        #x = self.noise2(x)
        x = self.nonlinear3(x)
        x = self.dropout2(x)
        #x = self.noise3(x)
        #x = self.batch_normalization5(x)
        x = self.nonlinear4(x)
        x = self.dropout6(x)
        #x = self.noise4(x)
        #x = self.batch_normalization3(x)
        
        #x = self.nonlinear4(x)
        x = self.nonlinear5(x)
        x = self.dropout4(x)
        #x = self.noise5(x)
        #x = self.dropout3(x)
        
        #x = self.noise6(x)
        #x = self.noise7(x)
        
        #x = self.nonlinear7(x)
        #x = self.dropout8(x)
        
        #x = self.batch_normalization5(x)
        #x = self.nonlinear8(x)
        #x = self.dense_add(x)
        #x = self.dropout4(x)
        x = self.add([x, shortcut])
        x = self.tanh(x)
        #x = self.dropout7(x)
        #x = self.nonlinear6(x)
        x = self.dropout9(x)
        x = self.output_layer(x)
        #x = self.add([x, shortcut])
        
        return x
        
class TDSE_NN_Conv_old(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, bias = False, dropout = 0.5, dropout_input = 0.6, activation = "tanh", conv_activation = "tanh"):
        """TDSE NN constructor"""
        super().__init__()
        #activation = tf.nn.relu
        self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(dropout_input)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(128, activation=activation, use_bias=bias)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(32, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(dropout)
        ### Fifth nonlinear dense layer
        self.nonlinear5 = tf.keras.layers.Dense(128, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout5 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        #self.nonlinear6 = tf.keras.layers.Dense(1, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout6 = tf.keras.layers.Dropout(dropout)
        self.nonlinear6 = tf.keras.layers.Dense(256, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout6 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear7 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout7 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear8 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout8 = tf.keras.layers.Dropout(dropout)
        self.nonlinear9 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout9 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        self.nonlinear10 = tf.keras.layers.Dense(32, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout10 = tf.keras.layers.Dropout(dropout)

        self.input_dense = tf.keras.layers.Dense(self.output_size, activation=activation)
        
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        
        self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias, activation="linear")
        #self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten(data_format = "channels_first") 
        self.flat = tf.keras.layers.Flatten() 
        ### Covolutional layer
        self.convolution1 = tf.keras.layers.Conv1D(8, 5)#, activation=activation)#, input_shape = (None, *self.input_shape_))
        self.convolution2 = tf.keras.layers.Conv1D(16, 5)#, activation=activation)#, input_shape = (None, *self.input_shape_))
        self.convolution3 = tf.keras.layers.Conv1D(32, 5)#, activation=activation)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))
        self.convolution4 = tf.keras.layers.Conv1D(64, 3)#, activation=activation)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))

        self.conv0 = tf.keras.layers.Conv1D(4, 3, activation=conv_activation)
        self.conv1 = tf.keras.layers.Conv1D(8, 3, activation=conv_activation)
        self.conv2 = tf.keras.layers.Conv1D(16, 3, activation=conv_activation)
        self.conv3 = tf.keras.layers.Conv1D(32, 3, activation=conv_activation)
        #self.max_pooling1 = tf.keras.layers.MaxPooling1D(5)
        self.max_pooling0 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling1 = tf.keras.layers.MaxPooling1D(3)
        #self.max_pooling2 = tf.keras.layers.MaxPooling1D(5)
        self.max_pooling2 = tf.keras.layers.MaxPooling1D(3)
        #self.max_pooling3 = tf.keras.layers.MaxPooling1D(5)
        self.max_pooling3 = tf.keras.layers.MaxPooling1D(3)
        self.max_pooling4 = tf.keras.layers.MaxPooling1D(3)
        self.avg_pooling1 = tf.keras.layers.AveragePooling1D(3)
        self.avg_pooling2 = tf.keras.layers.AveragePooling1D(3)
        self.avg_pooling3 = tf.keras.layers.AveragePooling1D(3)  
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization3 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization4 = tf.keras.layers.BatchNormalization()     
        self.batch_normalization5 = tf.keras.layers.BatchNormalization()     
        #self.batch_normalization5 = tf.keras.layers.BatchNormalization()     
        #self.batch_normalization5 = tf.keras.layers.BatchNormalization()     
        #self.batch_normalization5 = tf.keras.layers.BatchNormalization()   
        self.add = tf.keras.layers.Add()  
        self.tanh = tf.keras.layers.Activation("tanh")
        self.noise0 = tf.keras.layers.GaussianNoise(0.3)
        #self.noise1 = tf.keras.layers.GaussianNoise(0.1)
        self.noise1 = tf.keras.layers.GaussianNoise(0.3)
        self.noise2 = tf.keras.layers.GaussianNoise(0.1)
        self.noise3 = tf.keras.layers.GaussianNoise(0.1)
        self.noise4 = tf.keras.layers.GaussianNoise(0.1)
        self.noise5 = tf.keras.layers.GaussianNoise(0.1)
        self.noise6 = tf.keras.layers.GaussianNoise(0.1)
        #self.noise7 = tf.keras.layers.GaussianNoise(0.1)
        self.noise7 = tf.keras.layers.GaussianNoise(0.3)
        self.dense_add = tf.keras.layers.Dense(128)


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        #x = tf.transpose(x)
        #x = self.flatten_inp(x)
        #x = self.preprocess(x)
        #x_reshaped = tf.expand_dims(x, axis=2)
        
        #x = self.avg_pooling3(x_reshaped)
        #x = self.max_pooling1(x)
        #x = self.noise0(x)
        #x = self.conv0(x)
        #x = self.max_pooling0(x)
        #shortcut = self.flatten(x)
        #x = tf.expand_dims(x, axis=2)
        x = self.convolution1(x)

        #x = self.conv1(x)
        #x = self.avg_pooling1(x)
        x = self.max_pooling1(x)
        #x = self.conv2(x)
        #x = self.convolution2(x)
        #x = self.avg_pooling2(x)
        #x = self.max_pooling2(x)
        #x = self.conv3(x)
        #x = self.convolution3(x)
        #x = self.avg_pooling3(x)
        #x = self.max_pooling3(x)
        x = self.flatten(x)
        #x = self.flat(x)
        #x = self.batch_normalization4(x)
        #x = self.input_dense(x)
        #shortcut = x
        
        #x = self.dropout0(x)
        x = self.batch_normalization1(x)
        x = self.nonlinear1(x)
        shortcut = x
        x = self.noise1(x)
        x = self.nonlinear2(x)
        #x = self.dropout1(x)
        #x = self.batch_normalization2(x)
        x = self.noise2(x)
        x = self.nonlinear3(x)
        #x = self.dropout2(x)
        x = self.noise3(x)
        #x = self.batch_normalization5(x)
        x = self.nonlinear4(x)
        #x = self.dropout6(x)
        x = self.noise4(x)
        #x = self.batch_normalization3(x)
        
        #x = self.nonlinear4(x)
        x = self.nonlinear5(x)
        #x = self.dropout4(x)
        x = self.noise5(x)
        #x = self.dropout3(x)
        
        #x = self.noise6(x)
        #x = self.noise7(x)
        
        #x = self.nonlinear7(x)
        #x = self.dropout8(x)
        
        #x = self.batch_normalization5(x)
        #x = self.nonlinear8(x)
        #x = self.dense_add(x)
        #x = self.dropout4(x)
        x = self.add([x, shortcut])
        #x = self.tanh(x)
        #x = self.dropout7(x)
        #x = self.nonlinear6(x)
        #x = self.dropout9(x)
        x = self.output_layer(x)
        #x = self.add([x, shortcut])
        
        return x
        
    
class TDSE_NN_Conv_ResNet(tf.keras.Model):
    """TDSE NN model
    
    Model class - custom model for the source term computation.
    """    
    def __init__(self, input_shape_, output_size, bias = False, dropout = 0.5, dropout_input = 0.6, activation = "tanh"):
        """TDSE NN constructor"""
        super().__init__()
        #activation = tf.nn.relu
        self.output_size = output_size
        self.input_shape_ = input_shape_
        ### First input layer
        self.input_layer = tf.keras.layers.InputLayer(self.input_shape_)
        ### Add dropout for training to prevent overfitting
        self.dropout0 = tf.keras.layers.Dropout(dropout_input)
        ### Preprocessing layer
        #self.preprocess = Preprocessing(self.size)
        ### First nonlinear dense layer, array of size 256 as an output, input shape is the 
        ### size of the initial data
        self.nonlinear1 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)#, input_shape = (size,))
        ### Add dropout for training to prevent overfitting
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        ### Second nonlinear dense layer
        self.nonlinear2 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        ### Third nonlinear dense layer
        self.nonlinear3 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout3 = tf.keras.layers.Dropout(dropout)
        ### Fourth nonlinear dense layer
        self.nonlinear4 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout4 = tf.keras.layers.Dropout(dropout)
        ### Fifth nonlinear dense layer
        self.nonlinear5 = tf.keras.layers.Dense(64, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        #self.dropout5 = tf.keras.layers.Dropout(dropout)
        ### Sixth nonlinear dense layer
        #self.nonlinear6 = tf.keras.layers.Dense(1, activation=activation, use_bias=bias)
        ### Add dropout for training to prevent overfitting
        self.dropout6 = tf.keras.layers.Dropout(dropout)
        
        ### Simple linear output dense layer with linear activation function 
        ### f(y) = y, y = W.x + b
        
        self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias, activation="linear")
        #self.output_layer = tf.keras.layers.Dense(self.output_size, use_bias=bias)
        ### Final flattening layer
        self.flatten = tf.keras.layers.Flatten(data_format = "channels_first") 
        ### Covolutional layer
        self.convolution1 = tf.keras.layers.Conv1D(8, 3, activation=activation)#, input_shape = (None, *self.input_shape_))
        self.convolution2 = tf.keras.layers.Conv1D(16, 3, activation=activation)#, input_shape = (None, *self.input_shape_))
        #self.convolution3 = tf.keras.layers.Conv1D(32, 3)#, activation=tf.keras.activations.tanh)#, input_shape = (None, *self.input_shape_))
        self.avg_pooling1 = tf.keras.layers.MaxPooling1D(3)
        #self.avg_pooling1 = tf.keras.layers.AveragePooling1D(10)
        self.avg_pooling2 = tf.keras.layers.AveragePooling1D(3)
        self.avg_pooling3 = tf.keras.layers.AveragePooling1D(3)       

        self.add = tf.keras.layers.Add()
        self.dense_add = tf.keras.layers.Dense(self.output_size)


    def call(self, inputs):
        """TDSE NN call

        Specifies how to call the model.
        """
        x = self.input_layer(inputs)
        shortcut = self.flatten(x)
        #x = tf.transpose(x)
        #x = self.flatten_inp(x)
        #x = self.preprocess(x)
        #x_reshaped = tf.expand_dims(x, axis=2)
        #x_reshaped = tf.expand_dims(x, axis=2)
        
        #x = self.avg_pooling3(x_reshaped)
        x = self.convolution1(x)
        x = self.avg_pooling1(x)
        #x = self.convolution2(x)
        #x = self.avg_pooling2(x)
        x = self.flatten(x)
        
        x = self.dropout0(x)
        x = self.nonlinear1(x)
        x = self.dropout1(x)
        x = self.nonlinear2(x)
        x = self.dropout2(x)
        x = self.nonlinear3(x)
        x = self.dropout3(x)
        x = self.nonlinear4(x)
        x = self.dropout4(x)
        x = self.nonlinear5(x)
        x = self.dropout6(x)
        
        x = self.dense_add(x)
        x = self.add([x, shortcut])
        x = self.output_layer(x)
        
        return x
        
    