from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss
#from app.structure import data_transformer as DT
# from marshmallow import Schema, fields
from flask import abort

from marshmallow import Schema, fields, validate, ValidationError
from app.structure import model as md



class CreateDSSInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))
    training = fields.Int(required=True, validate=validate.Range(min=0, max=1))
    testing  = fields.Int(required=True, validate=validate.Range(min=0, max=1))


@app.route('/dss', methods=['POST'])
def dss_main():

    model = model_name = training = testing = None
    create_dss_schema = CreateDSSInputSchema()
    errors = create_dss_schema.validate(request.form)

    if errors:
        message = {
            'status': 404,
            'message': str(errors),
        }
        resp = jsonify(message)
        resp.status_code = 404
        return resp

    training = 0
    testing = 0
    if request.form.get('model_id'):
        model_id = int(request.form.get('model_id'))
    if request.form.get('training'):
        training = int(request.form.get('training'))
    if request.form.get('testing'):
        testing = int(request.form.get('testing'))

    if model_id:
        if (model_id == 1):
            DSS = dss.NeuralNetwork(app.config['NEURAL_NETWORK_MODEL'])
        elif (model_id == 2):
            DSS = dss.RandomForest(app.config['RANDOM_FOREST_CLASSIFIER_MODEL'])
        elif (model_id == 3):
            DSS = dss.LinearRegressionM(app.config['LINEAR_REGRESSION_MODEL'])
        elif (model_id == 4):
            DSS = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])

        data = None

        if training == 1:

            DSS.data_intialization()
            DSS.data_preprocessing()
            modelObject = DSS.training()

            if modelObject.get_model() is not None:

                DSS.save_model( modelObject.get_model() )
                data = {
                    'model_id': model_id,
                    'model_name': modelObject.get_name(),
                    'status': 200,
                    'message': 'Trained Successfully',
                    'results': [],
                    'accuracy': modelObject.get_accuracy()
                }
            else:

                data = {
                    'model_id': model_id,
                    'model_name': modelObject.get_name(),
                    'status': 500,
                    'message': 'Model is empty',
                    'results': [],
                    'accuracy': None
                }

        if testing == 1:
            model_reseponse = DSS.testing()
            data = {
                'model_id': model_id,
                'model_name': model_name,
                'status': 200,
                'message': 'Tested successully',
                'results': model_reseponse,
                'accuracy': None

            }

        resp = jsonify(data)
        resp.status_code = 200
        return resp

    else:
        return not_found()


@app.route('/cross-validation/k-fold', methods=['GET'])
def cross_validation_kfold_main():
    model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])
    accuracy = model.kfold()
    return 'awais'



@app.route('/amucad/api/find_amucad_objects/', methods=['GET'])
def find_amucad_objects():
    obj = DT.DataTransformer()
    return obj.get_data()


@app.route('/fish/training', methods=['GET'])
def fish_training():
    
    model_type = 1 #Random Forest
    
    #DSS model type is initialized
    mdObject = md.FdiAssessment(model_type)

    
    mdObject.start();

    return 'awais'


from app import errors