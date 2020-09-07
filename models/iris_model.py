from flask_restx import fields, Model
import numpy as np
from models.model import Model


class IrisPredictResult(Model):
    def __init__(self, predict_name):
        self.prediction_result = predict_name

    @classmethod
    def build_definition(cls, ns_iris):
        return ns_iris.model('PredictionResult', {
            'prediction_result': fields.String(description='Descrição da Iris classificada')
        })


class IrisPredictPayload(Model):
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width):
        self.__sepal_length = float(sepal_length)
        self.__sepal_width = float(sepal_width)
        self.__petal_length = float(petal_length)
        self.__petal_width = float(petal_width)

    def get_payload(self):
        return np.array([[self.__sepal_length, self.__sepal_width, self.__petal_length, self.__petal_width]])

    @classmethod
    def build_definition(cls, ns_iris):
        return ns_iris.model('PredictionPayload', {
            'sepal_length': fields.Float(description='Comprimento da sepala'),
            'sepal_width': fields.Float(description='Largura da sepala'),
            'petal_length': fields.Float(description='Comprimento da petala'),
            'petal_width': fields.Float(description='Largura da petala')
        })
