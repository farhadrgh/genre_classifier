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

data_train = np.load('../data_prep/43feature_proginprog/_train.npy', allow_pickle=True)
data_validation = np.load('../data_prep/43feature_proginprog/_validation.npy', allow_pickle=True)

# Data Preprocessing: Normalize data.
# Training set
X_train = np.concatenate(data_train[:,0])[:, :128, :20] #[:,0:128,:]
y_train = np.concatenate(data_train[:,1])

n_prog = len(y_train[y_train == 1])
n_nonprog = len(y_train[y_train == 0])
n_total = X_train.shape[0]
assert n_total == n_prog + n_nonprog

# handling data imbalanced
prog_ndx = np.where(y_train == 1)[0]
X_prog = X_train[prog_ndx, :, :]
assert X_prog.shape == (n_prog, X_train.shape[1], X_train.shape[2])

nonprog_ndx = np.where(y_train == 0)[0]
X_nonprog = X_train[nonprog_ndx, :, :]
assert X_nonprog.shape == (n_nonprog, X_train.shape[1], X_train.shape[2])


print (f'Number of Prog Songs = {n_prog}')
print (f'Number of NonProg Songs = {n_nonprog}')


# Validation set
X_validation = np.concatenate(data_validation[:,0])[:, :128, :20]#[:,0:128,:]
y_validation = np.concatenate(data_validation[:,1])

assert y_train.shape[1:] == y_validation.shape[1:]

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

print(f'X_validation shape: {X_validation.shape}')
print(f'y_validation shape: {y_validation.shape}')


# for trainable mean, std
mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)


# Layer classes
# Inherited to writout summary for tensorboard
class MyLSTM(LSTM):
    def call(self, inputs, mask=None, training=None, initial_state=None):
        ret = super(MyLSTM, self).call(inputs,
                                       mask=mask,
                                       training=training,
                                       initial_state=initial_state)
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
            shape=(1,1,20),
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
            shape=(1,1,20,),
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
        return input_shape[:self.axis]


# model parameters
batch_size = 256
n_epochs = 1000
input_shape = (X_train.shape[1], X_train.shape[2])

drop  = 0.2
rdrop = 0.2
adam = Adam(
    lr=0.001,
    decay=1.e-7
)
# Building the model
def NN():
    model = Sequential()
    model.add(MyLSTM(
        units=16,
        input_shape=input_shape, 
        kernel_initializer=keras.initializers.he_uniform(seed=None),
        dropout=drop,
        recurrent_dropout=rdrop,
        return_sequences=True,
        ))
    model.add(MyLSTM(
        units=8,
        kernel_initializer=keras.initializers.he_uniform(seed=None),
        dropout=drop,
        recurrent_dropout=rdrop,
        return_sequences=False,
        ))
    model.add(MyDense(units=4, activation='elu'))
    model.add(MyDense(units=1, activation='sigmoid'))

    return model

model = NN()
model.summary()


# Replicates `model` on 2 GPUs.
print("Compiling ...")

#parallel_model = multi_gpu_model(model, gpus=2)
model.compile(loss='binary_crossentropy',
                       optimizer=adam,
                       metrics=['accuracy'])

now = datetime.datetime.now()
tensorboard = keras.callbacks.TensorBoard(
    log_dir = f'./LSTM-{now.strftime("%Y%m%d_%H-%M")}' ,
    histogram_freq=1,
    batch_size=batch_size,
    write_graph=True)

# This `fit` call will be distributed on 2 GPUs.
# batch size is 1024, each GPU will process 512 samples.
print("Training ...")
model.fit(
    X_train,
    y_train,
    epochs=n_epochs,
    batch_size=batch_size,
    callbacks=[tensorboard],
    validation_data=(X_validation, y_validation)
    #validation_data=(X_validation, y_softmax_validation)
)


print("\nValidating ...")
score, accuracy = parallel_model.evaluate(X_validation, y_validation, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)
