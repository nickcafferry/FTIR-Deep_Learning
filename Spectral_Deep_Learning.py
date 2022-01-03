
#Import Library
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, losses, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Design function
def training_curve(hist, is_acc):
    history = hist.history
    fig, loss_ax = plt.subplots(figsize=(10, 6))
    loss_ax.plot(history['loss'], label='train loss')
    loss_ax.plot(history['val_loss'], label='val loss')
    if is_acc:
        loss_ax.plot(history['acc'], label='train acc')
        loss_ax.plot(history['val_acc'], label='val acc')
        loss_ax.plot(history['binary_accuracy'], label='train binacc')
        loss_ax.plot(history['val_binary_accuracy'], label='val binacc')
    loss_ax.set_xlabel('epoch')
    loss_ax.legend(loc='upper left')
    plt.show()
	
def acc(y_true, y_pred):
    compare = K.all(K.equal(y_true, K.round(y_pred)), axis=1)
    return K.mean(K.cast(compare, dtype='float32'))

#Load Data
## Preprocess Data
ir = pd.read_csv('ir.csv', index_col='Unnamed: 0').T
ir.reset_index(inplace = True)
interpolate_args = { "method" : "linear", "order" : 1 }
ir.iloc[:, 1:] = ir.iloc[:, 1:].interpolate(**interpolate_args, limit_direction='both', axis = 0)
ir.set_index('index', inplace = True)
ir = ir.div(ir.max(axis=0), axis=1)

ms = pd.read_csv('mass.csv', index_col='Unnamed: 0').T.fillna(0)
ms = ms.loc[:,ms.sum(axis=0)!=0]
ms = ms.div(ms.max(axis=0), axis=1)

target = pd.read_csv('target.csv', index_col='cas')

irms = pd.concat([ir, ms], axis=1, join='inner')
irms.index = irms.index.astype('int64')
df = pd.concat([irms, target], axis=1, join='inner')
df

## Split data
X_train, X_test, Y_train, Y_test = train_test_split(df[irms.columns], df[target.columns], test_size=0.2, random_state=7)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=77)

AE_train = irms.drop(X_test.index).drop(X_valid.index)
AE_valid = irms.loc[X_valid.index]
AE_test = irms.loc[X_test.index]

# Train AutoEncoder
inputlayer = layers.Input(shape=irms.shape[1])
encoded = layers.Dense(256, activation='relu')(inputlayer)
decoded = layers.Dense(irms.shape[1], activation='sigmoid')(encoded)

encoder = models.Model(inputlayer, encoded)
autoencoder = models.Model(inputlayer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
autoencoder.summary()

mc = ModelCheckpoint('AutoEncoder.h5', monitor='val_loss', mode='min', save_best_only=True)
hist = autoencoder.fit(AE_train, AE_train, epochs=500, validation_data=(AE_valid, AE_valid), batch_size=64, callbacks=[mc])
training_curve(hist, False)

# Load AutoEncoder
autoencoder = models.load_model('AutoEncoder.h5')
encoder = models.Model(autoencoder.layers[0].input, autoencoder.layers[1].output)
encoder.summary()

pred = autoencoder.predict(AE_test)
print(f"MAE : {mean_absolute_error(AE_test, pred)}")
print(f"MSE : {mean_squared_error(AE_test, pred)}")

#Train MLP Model

inputlayer = layers.Input(shape=256)
hiddenlayer = layers.Dense(200)(inputlayer)
hiddenlayer = layers.BatchNormalization()(hiddenlayer)
hiddenlayer = layers.ReLU()(hiddenlayer)
hiddenlayer = layers.Dropout(0.45)(hiddenlayer)
hiddenlayer = layers.Dense(150)(hiddenlayer)
hiddenlayer = layers.BatchNormalization()(hiddenlayer)
hiddenlayer = layers.ReLU()(hiddenlayer)
hiddenlayer = layers.Dropout(0.15)(hiddenlayer)
outputlayer = layers.Dense(target.shape[1], activation='sigmoid')(hiddenlayer)

model = models.Model(inputlayer, outputlayer)
model.compile(optimizer=Adam(learning_rate=0.00015), loss=losses.BinaryCrossentropy(), metrics=[acc, metrics.BinaryAccuracy()])
model.summary()

mc = ModelCheckpoint('Multi Layer Perceptron.h5', monitor='val_loss', mode='min', save_best_only=True)
hist = model.fit(encoder.predict(X_train), Y_train, epochs=500, batch_size=64, validation_data=(encoder.predict(X_valid), Y_valid), callbacks=[mc])
training_curve(hist, True)

model = models.load_model('Multi Layer Perceptron.h5')
pred = model.predict(encoder.predict(X_test)) > 0.5
print(f"Accuracy : {accuracy_score(Y_test, pred)}")
Y_test_flatten = Y_test.to_numpy().reshape(-1)
pred_flatten = pred.reshape(-1)
print(f"Binary Accuracy : {accuracy_score(Y_test_flatten, pred_flatten)}")
print(f"Balanced Binary Accuracy : {balanced_accuracy_score(Y_test_flatten, pred_flatten)}")
print(f"F1 score : {f1_score(Y_test_flatten, pred_flatten)}")
print("Confusion Matrix : ")
print(confusion_matrix(Y_test_flatten, pred_flatten))

#Train CNN Model
def get_Conv_Pool(n, layer, w_f=3, w_p=2):
    layer = layers.Conv1D(n, w_f)(layer)
    layer = layers.ELU()(layer)
    layer = layers.MaxPooling1D(w_p)(layer)
    return layer

def get_Dense(n, p, layer):
    layer = layers.Dense(n)(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.ELU()(layer)
    layer = layers.Dropout(p)(layer)
    return layer

inputlayer = layers.Input(shape=(irms.shape[1], 1))
hiddenlayer = get_Conv_Pool(4, inputlayer)
hiddenlayer = get_Conv_Pool(4, hiddenlayer)
hiddenlayer = get_Conv_Pool(8, hiddenlayer)
hiddenlayer = get_Conv_Pool(8, hiddenlayer)
hiddenlayer = get_Conv_Pool(16, hiddenlayer)
hiddenlayer = get_Conv_Pool(16, hiddenlayer)
hiddenlayer = layers.Flatten()(hiddenlayer)
hiddenlayer = get_Dense(128, 0.4, hiddenlayer)
hiddenlayer = get_Dense(128, 0.2, hiddenlayer)
outputlayer = layers.Dense(target.shape[1], activation='sigmoid')(hiddenlayer)

model = models.Model(inputlayer, outputlayer)
model.compile(optimizer=Adam(learning_rate=1e-3), loss=losses.BinaryCrossentropy(), metrics=[acc, metrics.BinaryAccuracy()])
model.summary()

mc = ModelCheckpoint('1D CNN.h5', monitor='val_loss', mode='min', save_best_only=True)
hist = model.fit(np.expand_dims(X_train, axis=-1), Y_train, epochs=200, batch_size=64, validation_data=(X_valid, Y_valid), callbacks=[mc])
training_curve(hist, True)

model = models.load_model('1D CNN.h5')
pred = model.predict(np.expand_dims(X_test, axis=-1)) > 0.5
print(f"Accuracy : {accuracy_score(Y_test, pred)}")
Y_test_flatten = Y_test.to_numpy().reshape(-1)
pred_flatten = pred.reshape(-1)
print(f"Binary Accuracy : {accuracy_score(Y_test_flatten, pred_flatten)}")
print(f"Balanced Binary Accuracy : {balanced_accuracy_score(Y_test_flatten, pred_flatten)}")
print(f"F1 score : {f1_score(Y_test_flatten, pred_flatten)}")
print("Confusion Matrix : ")
print(confusion_matrix(Y_test_flatten, pred_flatten))
