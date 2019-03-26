import pandas as pd
import keras
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

previssores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)

classificador = Sequential()
classificador.add(Dense(units=8,activation='relu', kernel_initializer='normal', input_dim=30))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=8,activation='relu', kernel_initializer='normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=16,activation='relu', kernel_initializer='normal'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=16,activation='relu', kernel_initializer='normal'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=1, activation='sigmoid'))
classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics = \
                      ['binary_accuracy'])

classificador.fit(previssores,classe, batch_size = 10, epochs=100)

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.014, 0.085, 0.0184,
                 158, 0.063]])

previssao = classificador.predict(novo)
resultados = cross_val_score(estimator = classificador,
                             X = previssores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()
