import math

from app import app
import os
from flask import Flask, request, jsonify, render_template

from app.input_schema import FDIInputSchema, CFInputSchema
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
    assessment_id = fields.Int(required=True, validate=validate.Range(min=1, max=10))


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
    mdObject = accuracy = assessment_name = model_name = None
    if assessment_id == 1:
        mdObject = md.FdiAssessment(model_type)
        accuracy = mdObject.start();
    elif assessment_id == 2:
        mdObject = md.CFAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 3:
        mdObject = md.LHIAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 4:
        mdObject = md.MuscelCWAAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 5:
        mdObject = md.LiverCWAAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 6:
        mdObject = md.ERYAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 7:
        mdObject = md.HBAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 8:
        mdObject = md.GLUAssessment(model_type)
        accuracy = mdObject.start()
    elif assessment_id == 9:
        mdObject = md.HCTAssessment(model_type)
        accuracy = mdObject.start()

    elif assessment_id == 10:
        mdObject = md.GillCWAAssessment(model_type)
        accuracy = mdObject.start()

    if mdObject is not None:
        assessment_name = mdObject.assessment_name
        model_name = mdObject.model_name
        message = 'Model trained successfully'
        status = 200
    else:
        message = 'something went wrong, please contact admin'
        status = 500

    message = {
        'status': status,
        'data': {
            'assessment': assessment_name,
            'model': model_name,
            'message': message,
            'accuracy': accuracy
        },
    }
    resp = jsonify(message)
    resp.status_code = 200
    return resp


@app.route('/finding/assessment', methods=['POST'])
def finding_assessment():
    data = {}
    data['assessment_id'] = int(request.form.get('assessment_id'))
    prediction = status = message = assessment_name = model_name = prediction_response = None

    if data['assessment_id'] == 1:
        validation = FDIInputSchema().validate(request.form)
        if validation:
            message = {
                'status': 422,
                'message': str(validation),
            }
            resp = jsonify(message)
            resp.status_code = 422
            return resp

        data['model_id'] = int(request.form.get('model_id'))

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

    elif data['assessment_id'] == 2:
        validation = CFInputSchema().validate(request.form)
        if validation:
            message = {
                'status': 422,
                'message': str(validation),
            }
            resp = jsonify(message)
            resp.status_code = 422
            return resp

        data['model_id'] = int(request.form.get('model_id'))
        data['Cryp1'] = int(request.form.get('Cryp1'))
        data['Cryp2'] = int(request.form.get('Cryp2'))
        data['Cryp3'] = int(request.form.get('Cryp3'))
        data['EpPap1'] = int(request.form.get('EpPap1'))
        data['EpPap2'] = int(request.form.get('EpPap2'))
        data['EpPap3'] = int(request.form.get('EpPap3'))
        data['FinRot'] = int(request.form.get('FinRot'))
        data['Locera1'] = int(request.form.get('Locera1'))
        data['Locera2'] = int(request.form.get('Locera2'))
        data['Locera3'] = int(request.form.get('Locera3'))
        data['PBT'] = int(request.form.get('PBT'))
        data['Skel1'] = int(request.form.get('Skel1'))
        data['Skel2'] = int(request.form.get('Skel2'))
        data['Skel3'] = int(request.form.get('Skel3'))
        data['Ulc1'] = int(request.form.get('Ulc1'))
        data['Ulc2'] = int(request.form.get('Ulc2'))
        data['Ulc3'] = int(request.form.get('Ulc3'))
        data['condition_factor'] = float(request.form.get('condition_factor'))
        mdObject = md.CFAssessment(model_type=data['model_id'])
        prediction = mdObject.predict_data(
            [[data['Cryp1'], data['Cryp2'], data['Cryp3'], data['EpPap1'], data['EpPap2'], data['EpPap3'],
              data['FinRot'], data['Locera1'], data['Locera2'], data['Locera3'], data['PBT'], data['Skel1'],
              data['Skel2'], data['Skel3'], data['Ulc1'], data['Ulc2'], data['Ulc3'], data['condition_factor']]])

    if prediction is not None:
        prediction_number = json.loads(prediction)[0]

        print(prediction_number)
        model_response_variable = json.loads(redis.Redis().get(mdObject.response_variable_key))
        status = 200

        print(prediction_number)
        print(model_response_variable)
        prd_response = model_response_variable[prediction_number]
        message = 'success'

        assessment_name = mdObject.assessment_name
        model_name = mdObject.model_name
        prediction_response = prd_response

    else:
        message = 'Please train model first.'
        status = 422

    message = {
        'status': status,
        'data': {
            'assessment': assessment_name,
            'model_name': model_name,
            'prediction': prediction_response,
            'message': message
        },
    }
    resp = jsonify(message)
    resp.status_code = status
    return resp

    print(data)


from app import errors
