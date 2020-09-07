from flask_restx import Namespace, Resource
from service.iris_classifier_service import IrisClassifierService

from models.iris_model import IrisPredictPayload, IrisPredictResult

print('\033[1;36m*' * 70)
print('*', '\t' * 4, 'Starting Iris Classifier Controller', '\t' * 4, '*')
print('\033[1;36m*\033[m' * 70)
iris_classifier_service = IrisClassifierService()

ns_iris = Namespace('iris')


@ns_iris.route('/classifier')
class Classifier(Resource):

    @ns_iris.doc('Classifier')
    @ns_iris.expect(IrisPredictPayload.build_definition(ns_iris))
    @ns_iris.marshal_with(IrisPredictResult.build_definition(ns_iris))
    def post(self):
        payload = IrisPredictPayload(**self.api.payload)
        return iris_classifier_service.predict(payload)


