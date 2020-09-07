import pandas as pd
import numpy as np
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, model_from_json

import os

from models.iris_model import IrisPredictPayload, IrisPredictResult


class IrisClassifierService:

    def __init__(self):
        self.ESTRUTURA_JSON_FILE_PATH = os.path.join(os.getcwd(), 'classificador_iris.json')
        self.WEIGHTS_FILE_PATH = os.path.join(os.getcwd(), 'iris_weigths.h5')

        self.__prepare_base()
        self.is_neural_network_ready_to_save = False
        self.__try_to_load_model()

    def predict(self, predict_payload: IrisPredictPayload):
        prediction = self.__classifier.predict(predict_payload.get_payload())
        prediction = (prediction > 0.5)
        prediction = prediction[0]

        label_index = np.argmax(prediction).item(0)
        iris = IrisPredictResult(self.classe_labels[label_index])
        return iris.to_json()

    def __prepare_base(self):
        base = pd.read_csv(os.path.join(os.getcwd(), 'iris.csv'))
        self.previsores = base.iloc[:, 0:4].values
        self.classe = base.iloc[:, 4].values
        self.classe_labels = []
        for c in self.classe:
            if c not in self.classe_labels:
                self.classe_labels.append(c)

        label_encoder = LabelEncoder()
        self.classe = label_encoder.fit_transform(self.classe)
        self.classe = np_utils.to_categorical(self.classe)

    def __build_model(self):
        print('\033[1;36m*' * 54)
        print('*', '\t' * 4, 'Building the Model', '\t' * 4, '*')
        print('\033[1;36m*\033[m' * 54)
        classificador = Sequential()
        classificador.add(Dense(units=4, kernel_initializer='random_uniform', activation='relu', input_dim=4))
        classificador.add(Dense(units=4, kernel_initializer='random_uniform', activation='relu'))
        classificador.add(Dense(units=3, activation='softmax'))

        classificador.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['categorical_accuracy'])

        return classificador

    def __train_model(self, model: Sequential):
        print('\033[1;36m*' * 54)
        print('*', '\t' * 4, 'Training the Model', '\t' * 4, '*')
        print('\033[1;36m*\033[m' * 54)
        model.fit(self.previsores, self.classe, batch_size=10, epochs=2000)
        self.is_neural_network_ready_to_save = True

    def save_model(self, model: Sequential):
        if not self.is_neural_network_ready_to_save:
            raise Exception('Could not save Neural Network, train model before save')

        print('\033[1;36m*' * 50)
        print('*', '\t' * 4, 'Saving the Model', '\t' * 4, '*')
        print('\033[1;36m*\033[m' * 50)
        json = model.to_json()
        with open(self.ESTRUTURA_JSON_FILE_PATH, 'w') as arquivo_json:
            arquivo_json.write(json)
            arquivo_json.close()

        model.save_weights(self.WEIGHTS_FILE_PATH)

    def __load_model(self) -> Sequential:
        print('\033[1;36m*' * 50)
        print('*', '\t' * 4, 'Loading the Model', '\t' * 4, '*')
        print('\033[1;36m*\033[m' * 50)

        arquivo_json = open(self.ESTRUTURA_JSON_FILE_PATH, 'r')
        estrutura = arquivo_json.read()
        arquivo_json.close()

        classificador = model_from_json(estrutura)
        classificador.load_weights(self.WEIGHTS_FILE_PATH)

        return classificador

    def __try_to_load_model(self):
        try:
            self.__classifier = self.__load_model()
        except FileNotFoundError:
            self.__classifier = self.__build_model()
            self.__train_model(self.__classifier)
            self.save_model(self.__classifier)
