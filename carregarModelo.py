from keras.models import model_from_json
import numpy as np
import pandas as pd

arq = open('classificador_breast.json','r')
estrutura = arq.read()
arq.close()

classificador = model_from_json(estrutura)
classificador.load_weights('classificador_breast.h5')

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.014, 0.085, 0.0184,
                 158, 0.063]])

previssao = classificador.predict(novo)
previssao_b = (previssao > 0.5)

previssores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

resultado = classificador.evaluate(previssores,classe)