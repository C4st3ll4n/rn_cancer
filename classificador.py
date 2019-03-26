import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score
import keras

entrada = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

previssores_treinamento, previssores_teste, classe_treinamento, classe_teste = train_test_split(entrada, classe, train_size=0.7)
    
classificador = Sequential()
classificador.add(Dense(units=16,activation='relu', kernel_initializer='random_uniform', input_dim=30))

classificador.add(Dense(units=16,activation='relu', kernel_initializer='random_uniform'))

classificador.add(Dense(units=1, activation='sigmoid'))

otimizador = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)

classificador.compile(optimizer=otimizador, loss='binary_crossentropy', metrics = \
                      ['binary_accuracy'])

classificador.fit(previssores_treinamento, classe_treinamento, batch_size = 10\
                  ,epochs=100)

pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsores = classificador.predict(previssores_teste)
previsores = (previsores > 0.5)

precissao = accuracy_score(classe_teste, previsores)
matrix = confusion_matrix(classe_teste, previsores)
resultado = classificador.evaluate(previssores_teste, classe_teste)
#print("DataFrame: {}".format(entrada))