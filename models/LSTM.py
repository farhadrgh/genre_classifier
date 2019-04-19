import keras
import datetime
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K

K.tensorflow_backend._get_available_gpus()
print(f"{64 * '#'} \n")

data_train = np.load('../data_prep/_train/_train.npy')
data_validation = np.load('../data_prep/_validation/_validation.npy')

X_train = np.concatenate(data_train[:,0])[:, :, :13] #[:,0:128,:]
y_train = np.concatenate(data_train[:,1])

X_validation = np.concatenate(data_validation[:,0])[:, :, :13]#[:,0:128,:]
y_validation = np.concatenate(data_validation[:,1])

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')

print(f'X_validation shape: {X_validation.shape}')
print(f'y_validation shape: {y_validation.shape}')

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_train = scaler.fit_transform(np.array(X_train[:, :, :], dtype = float))
#X_validation = scaler.fit_transform(np.array(X_train[:, :, :], dtype = float))

# LSTM model parameters
batch_size = X_train.shape[0]
n_epochs = 2000
input_shape = (X_train.shape[1], X_train.shape[2])

drop  = 0.35
rdrop = 0.05
adam = Adam(lr=0.01)


# Building the model
def NN():
    model = Sequential()
    model.add(LSTM(units=256, dropout=drop, recurrent_dropout=rdrop, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=64, dropout=drop, recurrent_dropout=rdrop, return_sequences=False))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=4, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    return model

model = NN()
model.summary()


# Replicates `model` on 2 GPUs.
print("Compiling ...")
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='binary_crossentropy',
                       optimizer=adam,
                       metrics=['accuracy'])


# This `fit` call will be distributed on 2 GPUs.
# batch size is 1024, each GPU will process 512 samples.
print("Training ...")
parallel_model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)


print("\nValidating ...")
score, accuracy = parallel_model.evaluate(X_validation, y_validation, batch_size=batch_size, verbose=1)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


dummy_model = NN()
dummy_model.set_weights(parallel_model.get_weights())

print("Saving the model ...")
now = datetime.datetime.now()
dummy_model.save(f'./logs/{now.strftime("%Y%m%d_%H-%M")}.h5')

#print("\nTesting ...")
#score, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
#print("Test loss:  ", score)
#print("Test accuracy:  ", accuracy)
