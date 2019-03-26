import pandas as pd
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np

base = pd.read_csv('../iris/iris.csv')
previssores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values
lcoder = LabelEncoder()
classe = lcoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def createNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units= neurons, input_dim=4, activation=activation,
                           kernel_initializer=kernel_initializer ))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units= neurons, activation=activation, kernel_initializer=
                            kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation='softmax'))
    classificador.compile(optimizer=optimizer,loss=loss, metrics=['accuracy'])
    return classificador


classificador = KerasClassifier(build_fn=createNetwork)

parametros = {'batch_size': [10,30], 'epochs': [5, 10], 
              'optimizer': ['adam', 'sgd'],
              'loss': ['sparse_categorical_crossentropy',], 
              'kernel_initializer':['random_uniform','normal'], 
              'activation': ['relu', 'tanh'], 'neurons':[8, 4]}

grid = GridSearchCV(estimator=classificador, param_grid=parametros,cv=2)


c_teste2 = [np.argmax(t) for t in classe_dummy ]
previsoes2 = [np.argmax(t) for t in previssores ]

grid = grid.fit(previssores, classe)
melhores_parametros = grid.best_params_
melhor_precissao = grid.best_score_

classificador_json = grid.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('classificador_iris.h5')