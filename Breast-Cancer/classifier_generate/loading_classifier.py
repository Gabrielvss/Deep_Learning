 
import numpy as np
from keras.models import model_from_json

#lendo o arquivo da rede
arquivo = open('breast_classifier.json','r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
#lendo o arquivo dos pesos
classificador.load_weights('weights_breast_classifier.h5')


#testando em um novo valor de entrada 
nova_entrada = np.array([[55.80, 9.45, 132.56, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.2,
                         0.05, 1099, 0.87, 4500, 145.2, 2, 0.04, 0.05, 0.015, 0.03, 0.007,
                         23.15, 16.66, 178, 2020, 0.14, 0.185, 0.88, 155, 45]])

previsao = classificador.predict(nova_entrada)
print(previsao)

#simulando um grande teste com os mesmos dados 
previsores = pd.read_csv('../entradas-breast.csv')
classe = pd.read_csv('../saidas-breast.csv')
classificador.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
resultado = classificador.evaluate(previsores, classe)
print(resultado)

