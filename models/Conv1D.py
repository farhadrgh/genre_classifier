import keras
import datetime
import numpy as np
import tensorflow as tf
import sklearn

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K

#K.tensorflow_backend._get_available_gpus()
#print(f"{64 * '#'} \n")

data_train = np.load('../data_prep/43feature/_train.npy', allow_pickle=True)
data_validation = np.load('../data_prep/43feature/_validation.npy', allow_pickle=True)

# Data Preprocessing: Normalize data.
# Training set
X_train = np.concatenate(data_train[:,0])#[:, :, :40] #[:,0:128,:]
y_train = np.concatenate(data_train[:,1])

y_train = np.repeat(y_train, X_train.shape[1], axis=-1)[:,:,np.newaxis] # fix for categorical
#y_softmax_train = np.concatenate([y_train, np.abs(y_train-1)], axis=-1)

mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)

# Validation set
X_validation = np.concatenate(data_validation[:,0])#[:, :, :40]#[:,0:128,:]
y_validation = np.concatenate(data_validation[:,1])

y_validation = np.repeat(y_validation, X_validation.shape[1], axis=-1)[:,:,np.newaxis]
#y_softmax_validation = np.concatenate([y_validation, np.abs(y_validation-1)], axis=-1)

assert y_train.shape[1:] == y_validation.shape[1:]

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

print(f'X_validation shape: {X_validation.shape}')
print(f'y_validation shape: {y_validation.shape}')

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = scaler.fit_transform(np.array(X_train[:, :, :], dtype = float))
#X_validation = scaler.fit_transform(np.array(X_train[:, :, :], dtype = float))

#X_train = (X_train - mean) / std
#X_validation = (X_validation - mean) / std

# model parameters
batch_size = 256 #X_train.shape[0]
n_epochs = 2000
input_shape = (X_train.shape[1], X_train.shape[2])

drop  = 0.35
rdrop = 0.05
adam = Adam(
    lr=0.001,
    decay=1.e-7
)


# Layer classes
# Inherited to writout summary for tensorboard
class MyLSTM(LSTM):
    def call(self, inputs, mask=None, training=None, initial_state=None):
        ret = super(MyLSTM, self).call(inputs, mask=mask, training=training, initial_state=initial_state)
        activation = ret
        tf.summary.histogram(
            'activation',
            activation)
        return activation

class MyDense(Dense):
    def call(self, inputs):
        activation = super(MyDense, self).call(inputs)
        tf.summary.histogram(
            'activation',
            activation)
        return activation

class Subtract(keras.layers.Layer):
    def __init__(self, value=0.0, **kwargs):
        self.init_value = np.float32(value)
        super(Subtract, self).__init__(**kwargs)

    def build(self, input_shape):
        self.value = self.add_weight(
            name="value",
            shape=(1,1,43),
            initializer=keras.initializers.Constant(
                value=np.float32(self.init_value)),
            trainable=True)
        super(Subtract, self).build(input_shape)
    def call(self, inputs):
        return inputs - self.value

class Multiply(keras.layers.Layer):
    def __init__(self, value=1.0, **kwargs):
        self.init_value = np.float32(value)
        super(Multiply, self).__init__(**kwargs)
    def build(self, input_shape):
        self.log_value = self.add_weight(
            name="log_value",
            shape=(1,1,43,),
            initializer=keras.initializers.Constant(
                value=np.log(np.float32(self.init_value))),
            trainable=True)
        super(Multiply, self).build(input_shape)
    def call(self, inputs):
        return inputs * tf.exp(self.log_value)

class Conv1D(keras.layers.Conv1D):
    def call(self, inputs):
        activation = super(Conv1D, self).call(inputs)
        tf.summary.histogram(
            'activation',
            activation)
        return activation

class Squeeze(keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Squeeze, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Squeeze, self).build(input_shape)
    def call(self, inputs):
        return keras.backend.squeeze(inputs, self.axis)
    def compute_output_shape(self, input_shape):
        #return [input_shape[0], input_shape[1]]
        return input_shape[:self.axis]

# Building the model
def NN():
    model = Sequential()
    #model.add(tf.keras.layers.InputLayer(input_shape=(None,256,13)))
    model.add(Subtract(mean, input_shape=input_shape))
    model.add(Multiply(1/std))
    #model.add(MyLSTM(units=256, dropout=drop, recurrent_dropout=rdrop, return_sequences=True, input_shape=input_shape))
    #model.add(MyLSTM(units=64, dropout=drop, recurrent_dropout=rdrop, return_sequences=False))
    
    for layer_idx in range(3):
        model.add(Conv1D(
            filters=256,#*(1+layer_idx),
            kernel_size=3,
            strides=1,
            dilation_rate=(3**layer_idx),
            padding="same",
            #data_format="channels_last",
            activation='elu',
            input_shape=input_shape))
        model.add(keras.layers.SpatialDropout1D(
            rate=0.1
        ))
    model.add(Conv1D(
        filters=1,
        kernel_size=1,
        strides=1,
        dilation_rate=1,
        padding="same",
        #data_format="channels_last",
        activation='sigmoid'))
    #model.add(Squeeze(axis=-1))
#    model.add(keras.layers.GlobalAveragePooling1D())
#    model.add(MyDense(units=64, activation='elu'))
#    model.add(MyDense(units=4, activation='elu'))
#    model.add(MyDense(units=1, activation='sigmoid'))
    
    return model

model = NN()
model.summary()


# Replicates `model` on 2 GPUs.
print("Compiling ...")
#parallel_model = multi_gpu_model(model, gpus=2)
model.compile(loss='binary_crossentropy',
                       optimizer=adam,
                       metrics=['accuracy'])

tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logscnn_43feats_single_learnable_norm',
    histogram_freq=1,
    batch_size=batch_size,
    write_graph=True)

# This `fit` call will be distributed on 2 GPUs.
# batch size is 1024, each GPU will process 512 samples.
print("Training ...")
model.fit(
    X_train,
    y_train,
    #y_softmax_train,
    epochs=n_epochs,
    batch_size=batch_size,
    callbacks=[tensorboard],
    validation_data=(X_validation, y_validation)
    #validation_data=(X_validation, y_softmax_validation)
)


print("\nValidating ...")
score, accuracy = parallel_model.evaluate(X_validation, y_validation, batch_size=batch_size, verbose=1)
#score, accuracy = model.evaluate(X_validation, y_softmax_validation, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


#dummy_model = NN()
#dummy_model.set_weights(parallel_model.get_weights())


print("Saving the model ...")
now = datetime.datetime.now()
#dummy_model.save(f'./weights/{now.strftime("%Y%m%d_%H-%M")}.h5')

#print("\nTesting ...")
#score, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
#print("Test loss:  ", score)
#print("Test accuracy:  ", accuracy)
