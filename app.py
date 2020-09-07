from flask import Flask
from flask_restx import Api
from controllers.iris_classifier_controller import ns_iris

app = Flask(__name__)
api = Api(app=app, version='1.0', title='REST-IRIS', description='A simple iris-classification API')

api.add_namespace(ns_iris)

if __name__ == '__main__':
    app.run()
