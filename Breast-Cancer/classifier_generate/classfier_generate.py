import pandas as pd
import keras
from keras.models import Sequential 
from keras.layers import Dropout, Dense


df_input = pd.read_csv('entradas-breast.csv')
df_output = pd.read_csv('saidas-breast.csv')


classificador = Sequential()
classificador.add(Dense(units=16,activation='relu',
kernel_initializer='random_uniform', input_dim=30
))
#adcionando Dropout a camada de entrada
classificador.add(Dropout(0.2))
#nova camada
classificador.add(Dense(units=16,activation='relu',
kernel_initializer='random_uniform'
))
 #adcionando Dropout a camada oculta
classificador.add(Dropout(0.2))
#camada de saida
classificador.add(Dense(units=1, activation='sigmoid'))
adam = keras.optimizers.Adam(learning_rate=0.001,decay=0.0001,clipvalue=0.5)
classificador.compile(optimizer= 'adam', loss='binary_crossentropy',metrics=['binary_accuracy'])
#treinando o classificador
classificador.fit(df_input,df_output,batch_size=10,epochs=100)

classificador_json = classificador.to_json()

#salvando os parametros da rede como json
with open('breast_classifier.json', 'w') as json_file:
    json_file.write(classificador_json)
    
#salvando os pesos da rede
classificador.save_weights('weigths_breast_classifier.h5')    