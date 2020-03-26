import math

from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss
# from app.structure import data_transformer as DT
# from marshmallow import Schema, fields
from flask import abort

from marshmallow import Schema, fields, validate, ValidationError
from app.structure import model as md
import redis
import json


class CreateDSSInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))
    training = fields.Int(required=True, validate=validate.Range(min=0, max=1))
    testing = fields.Int(required=True, validate=validate.Range(min=0, max=1))


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

                DSS.save_model(modelObject.get_model())
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


class TrainingAPISchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))
    assessment_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))


@app.route('/fish/training', methods=['POST'])
def fish_training():
    errors = TrainingAPISchema().validate(request.form)
    if errors:
        message = {
            'status': 422,
            'message': str(errors),
        }
        resp = jsonify(message)
        resp.status_code = 422
        return resp
    model_type = int(request.form.get('model_id'))  # Random Forest
    assessment_id = int(request.form.get('assessment_id'))
    # DSS model type is initialized
    if assessment_id == 1:
        mdObject = md.FdiAssessment(model_type)
        mdObject.start();
    return 'awais'


# https://github.com/marshmallow-code/marshmallow

def validate_sex(n):
    if n != 'm' and n != 'w' and n != 'n':
        raise ValidationError("Sex should be m,w or n")


def validate_group(n):
    if n != 'EXT' and n != 'LEEXT':
        raise ValidationError("Group should be EXT or LEEXT")


class CreateRTInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))
    assessment_id = fields.Int(required=True, validate=validate.Range(min=1, max=10))

    station = fields.Int(required=True)
    year = fields.Int(required=True)
    month = fields.Int(required=True)
    day = fields.Int(required=True)
    group = fields.Str(required=True, validate=validate_group)
    sex = fields.Str(required=True, validate=validate_sex)
    fish_no = fields.Int(required=True)
    total_length = fields.Int(required=True)
    total_weight = fields.Int(required=True)
    latitude = fields.Float(required=True)
    longitude = fields.Float(required=True)
    bottom_temperature = fields.Float(required=True)
    bottom_salinity = fields.Float(required=True)
    bottom_oxygen_saturation = fields.Float(required=True)
    hydrography_depth = fields.Float(required=True)
    fdi = fields.Float(required=True)


@app.route('/finding/assessment', methods=['POST'])
def finding_assessment():
    data = {}
    errors = CreateRTInputSchema().validate(request.form)
    if errors:
        message = {
            'status': 422,
            'message': str(errors),
        }
        resp = jsonify(message)
        resp.status_code = 422
        return resp

    data['model_id'] = int(request.form.get('model_id'))
    data['assessment_id'] = int(request.form.get('assessment_id'))

    data['station'] = int(request.form.get('station'))
    data['year'] = int(request.form.get('year'))
    data['month'] = int(request.form.get('month'))
    data['day'] = int(request.form.get('day'))

    data['sex'] = request.form.get('sex')
    data['group'] = request.form.get('group')

    data['fish_no'] = request.form.get('fish_no')
    data['total_length'] = int(request.form.get('total_length'))
    data['total_weight'] = int(request.form.get('total_weight'))
    data['latitude'] = float(request.form.get('latitude'))
    data['longitude'] = float(request.form.get('longitude'))
    data['bottom_temperature'] = float(request.form.get('bottom_temperature'))
    data['bottom_salinity'] = float(request.form.get('bottom_salinity'))
    data['bottom_oxygen_saturation'] = float(request.form.get('bottom_oxygen_saturation'))
    data['hydrography_depth'] = float(request.form.get('hydrography_depth'))
    data['fdi'] = float(request.form.get('fdi'))

    if data['assessment_id'] == 1:
        sex_m = sex_n = sex_w = 0
        if data['sex'] == 'm':
            sex_m = 1
        elif data['sex'] == 'n':
            sex_n = 1
        elif data['sex'] == 'w':
            sex_w = 1

        group_EXT = group_LEEXT = 0
        if data['group'] == 'EXT':
            group_EXT = 1
        if data['group'] == 'LEEXT':
            group_LEEXT = 1

        mdObject = md.FdiAssessment(model_type=data['model_id'])
        prediction = mdObject.predict_data([[data['station'], data['year'], data['month'], data['day'], data['fish_no'],
                                             data['total_length'], data['total_weight'], data['latitude'],
                                             data['longitude'],
                                             data['bottom_temperature'],
                                             data['bottom_salinity'], data['bottom_oxygen_saturation'],
                                             data['hydrography_depth'],
                                             data['fdi'],
                                             sex_m, sex_n, sex_w, group_EXT, group_LEEXT]])

        prd_response = status = message = None
        if prediction is not None:
            prediction_number = json.loads(prediction)[0]
            number_dec = prediction_number - int(prediction_number)
            if number_dec > 0.75:
                prediction_number = math.ceil(prediction_number)
            else:
                prediction_number = math.floor(prediction_number)
            model_response_variable = json.loads(redis.Redis().get(mdObject.response_variable_key))
            status = 200
            prd_response = model_response_variable[prediction_number]
            message = 'success'
        else:
            message = 'Please train model first.'
            status = 422

        message = {
            'status': status,
            'data': {
                'assessment': mdObject.assessment_name,
                'model': mdObject.model_name,
                'prediction': prd_response,
                'message': message
            },
        }
        resp = jsonify(message)
        resp.status_code = status
        return resp
    print(data)


from app import errors
