import json
from functools import wraps

import datetime
import jwt
import os
import redis
import requests
from flasgger import Swagger
from flask import request, jsonify

from app import app
from app.input_schema import FDIInputSchema, CFInputSchema, TrainingAPISchema, MunitionInputSchema, LoginInputSchema
from app.structure import dss
from app.structure import model as md, data_transformer as amc
import sys

swagger = Swagger(app, template=app.config['SWAGGER_TEMPLATE'])


# https://www.youtube.com/watch?v=J5bIPtEbS0Q
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'token' not in request.args:
            message = {
                'status': 403,
                'message': 'Token is missing',
            }
            resp = jsonify(message)
            resp.status_code = 403
            return resp
        try:
            token = request.args.get('token')
            jwt.decode(token, app.config['SECRET_KEY'])

        except:
            message = {
                'status': 403,
                'message': 'Invalid Token',
            }
            resp = jsonify(message)
            resp.status_code = 403
            return resp
        return f(*args, **kwargs)

    return decorated


@app.route('/cross-validation/k-fold', methods=['GET'])
def cross_validation_kfold_main():
    model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])
    accuracy = model.kfold()
    return 'awais'


@app.route('/amucad/api/data_transformation/', methods=['GET'])
@token_required
def find_amucad_objects():
    """Transforming AMUCAD data into CSV format
    This is using docstrings for specifications.
    ---
    parameters:
      - name: token
        in: query
        type: string
        required: true
        description:  ''

    responses:
      200:
        description:
        examples:
          rgb: []
    """
    obj = amc.Amucad()
    obj.transform_objects_to_csv()
    return 'true'


@app.route('/dss/training', methods=['POST'])
def fish_training():
    """Endpoint for training dss system
    This is using docstrings for specifications.
    ---
    parameters:
      - name: model_id
        in: formData
        type: integer
        enum: [1, 2, 3, 5, 6]
        required: true
        default: 1
        description:  1 => Neural Network, 2 => RANDOM FOREST, 3 => LINEAR REGRESSION, 5=> DEEP NEURAL NETWORK, , 6=> Decision Tree
      - name: action_id
        in: formData
        type: integer
        required: true
        default: 1
      - name: protection_goods_id
        in: formData
        type: integer
        required: true
        default: 2
      - name: user_id
        in: formData
        type: integer
        required: true
        default: 2
    responses:
      200:
        description: JSON object containing status of the action
        examples:
          rgb: []
    """
    errors = TrainingAPISchema().validate(request.form)
    if errors:
        message = {
            'status': 422,
            'message': str(errors),
        }
        resp = jsonify(message)
        resp.status_code = 422
        return resp
    model_type = int(request.form.get('model_id'))
    action_id = int(request.form.get('action_id'))
    protection_goods_id = int(request.form.get('protection_goods_id'))
    user_id = int(request.form.get('user_id'))

    from app.structure import machine_learning as starter
    try:
        obj = starter.MachineLearning(model_type, action_id, protection_goods_id, user_id)
        res = obj.process()
        status = res['status']
        ret = res['message']

    except Exception as e:
        status = 500
        ret = str(e)

    message = {
        'status': status,
        'data': {
            'message': ret
        },
    }
    resp = jsonify(message)
    resp.status_code = status
    return resp


@app.route('/finding/assessment', methods=['POST'])
def finding_assessment():
    data = {'assessment_id': int(request.form.get('assessment_id'))}
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

        mdObject = md.FdiAssessment(model_type=data['model_id'])
        return mdObject.getFdiAssessment(data)

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
        return mdObject.getCFAssessment(data)
    return ''


@app.route('/login', methods=['POST'])
def login():
    """Endpoint to get the token for authentication
    This is using docstrings for specifications.
    ---
    parameters:
      - name: email
        in: formData
        type: string
        required: true
        description:  ''

      - name: password
        in: formData
        type: string
        format: password
        required: true
        description:  ''
    responses:
      200:
        description: A JSON object containing token to access further endpoints
        examples:
          rgb: []
    """
    validation = LoginInputSchema().validate(request.form)
    if validation:
        resp = jsonify({
            'status': 422,
            'message': str(validation),
        })
        resp.status_code = 422
        return resp
    r_login = requests.post('https://mdb.in.tu-clausthal.de/api/v1/auth/login',
                            data={'email': request.form.get('email'), 'password': request.form.get('password')}).json()
    if r_login['status'] == 200:
        token = jwt.encode(
            {'user': request.form.get('email'), 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)},
            app.config['SECRET_KEY'])
        resp = jsonify({
            'status': 200,
            'token': token.decode('UTF-8'),
        })
        resp.status_code = 200
    else:
        resp = jsonify({
            'status': 422,
            'message': 'Incorrect email or password',
        })
        resp.status_code = 422
    return resp


@app.route('/dss/evaluation', methods=['POST'])
def dss_evaluation():
    """Endpoint to get the token for authentication
    This is using docstrings for specifications.
    ---
    parameters:
      - name: Input
        in: body
        type: string
        required: true
        description:  ''

    responses:
      200:
        description: Array of assessment objects
    """
    try:
        content = request.get_json(silent=True)
        from app.structure import machine_learning as starter

        results = []
        counter = 1
        model_key = 'model_id'
        action_key = 'action_id'
        protection_key = 'protection_good_id'

        status = 200
        if content is not None:
            counter = 1
            for d in content:
                status = 400
                single_result = {}
                if model_key not in d or action_key not in d or protection_key not in d:
                    single_result[
                        'assessment_' + str(counter)] = "Action ID, Model ID and Protection Good ID must be provided."
                    single_result['status'] = status
                else:
                    if 0 < d['model_id'] < 7:
                        obj = starter.MachineLearning(d['model_id'], d['action_id'], d['protection_good_id'])
                        single_result['model_id'] = d['model_id']
                        single_result['protection_good_id'] = d['protection_good_id']
                        single_result['action_id'] = d['action_id']
                        sample = []
                        for key in d['data'][0]:
                            sample.append(d['data'][0][key])

                        obj.set_test_data(sample)
                        p_r = obj.testing()
                        if p_r is not None:
                            status = 200
                            single_result['assessment_response'] = obj.testing()
                            single_result['status'] = status
                        else:
                            status = 404
                            single_result['assessment_response'] = 'Model Does Not Exist'
                            single_result['status'] = status
                    else:
                        status = 404
                        single_result['assessment_response'] = 'Model_id should be between 1 and 6'
                        single_result['status'] = status

                counter = counter + 1
                results.append(single_result)
        else:
            status = 400
            single_result = {'status': status, 'message': 'Invalid Json Input'}
            results.append(single_result)

    except Exception as e:
        status = 500
        results.append({'message':str(e)})

    resp = jsonify(results)
    resp.status_code = status
    return resp


@app.route('/dss/logs', methods=['GET'])
def dss_logs():
    """Endpoint to get the token for authentication
    This is using docstrings for specifications.
    ---
    responses:
      200:
        description: Array of logs
    """
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    collection_training = client['dss']['training_history']
    mg_data = collection_training.find()
    data = []
    for post in mg_data:
        data.append({
            'user_id': post['user_id'],
            'model_name': post['model_name'],
            'model_id': post['model_id'],
            'action_id': post['action_id'],
            'protection_goods_id': post['protection_goods_id'],
            'training_observations': post['training_observations'],
            'input_features': post['input_features'],
            'output_variables': post['output_variables'],
            'date': post['date']
        })

    message = {
        'status': 200,
        'data': data
    }
    resp = jsonify(message)
    resp.status_code = 200
    return resp
