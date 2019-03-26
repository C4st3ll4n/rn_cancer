
import pandas as pd
import keras

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

entrada = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')


def createNetwork(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units=1, activation='sigmoid'))

    classificador.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    
    return classificador


classificador = KerasClassifier(build_fn=createNetwork)

parametros = {'batch_size': [10,30], 'epochs': [50, 100], 
              'optimizer': ['adam', 'sgd'],
              'loss': ['binary_crossentropy','hinge'], 
              'kernel_initializer':['random_uniform','normal'], 
              'activation': ['relu', 'tanh'], 'neurons':[16, 8]}

grid = GridSearchCV(estimator=classificador, param_grid=parametros,
                    scoring='accuracy', cv=5)

grid = grid.fit(entrada, classe)
melhores_parametros = grid.best_params_
melhor_precissao = grid.best_score_
