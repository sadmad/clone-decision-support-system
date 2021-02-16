import json
from functools import wraps

import datetime
import jwt
import requests
from flasgger import Swagger
from flask import request, jsonify

from app import app
from app.input_schema import TrainingAPISchema, LoginInputSchema

swagger = Swagger(app, template=app.config['SWAGGER_TEMPLATE'])


# https://www.youtube.com/watch?v=J5bIPtEbS0Q
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'token' in request.args or request.form.get('token') is not None:

            try:
                token = request.args.get('token') if 'token' in request.args else request.form.get('token')
                jwt.decode(token, app.config['SECRET_KEY'])

            except:
                message = {
                    'status': 403,
                    'message': 'Invalid Token',
                }
                resp = jsonify(message)
                resp.status_code = 403
                return resp
        else:
            message = {
                'status': 403,
                'message': 'Token is missing',
            }
            resp = jsonify(message)
            resp.status_code = 403
            return resp

        return f(*args, **kwargs)

    return decorated


@app.route('/dss/training', methods=['POST'])
@token_required
def dss_training():
    """Endpoint for training dss system for munitions
    This is using docstrings for specifications.
    ---
    parameters:
      - name: model_id
        in: formData
        type: integer
        enum: [1, 2, 3, 5, 6]
        required: true
        default: 1
        description:  User can choose any of the given model for training. System will override the existing model for selected model_id, action_id and protection_goods_id. Neural Network(scikit learn)=1, RANDOM FOREST(scikit learn)=2, LINEAR REGRESSION(scikit learn)=3, DEEP NEURAL NETWORK(Keras+tensorflow)=5, Decision Tree(scikit learn)=6
      - name: action_id
        in: formData
        type: integer
        required: true
        description: Value from AMUCAD application. Possible values corrosion=2, explosion=1
        default: 1
      - name: protection_goods_id
        in: formData
        type: integer
        required: true
        description: Value from AMUCAD application. Possible values 1,2,3,4,5
        default: 2
      - name: user_id
        in: formData
        type: string
        required: true
        description: 'User id of logged in user in AMUCAD application should be given'
        default: 2
      - name: token
        in: formData
        type: string
        required: true
        description:  'JSON Web Token, should be generated from login API using email and password'
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
    user_id = request.form.get('user_id')

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


@app.route('/dss/metrics', methods=['POST'])
@token_required
def dss_accuracy():
    """Endpoint for training dss system for munitions
    This is using docstrings for specifications.
    ---
    parameters:
      - name: model_id
        in: formData
        type: integer
        enum: [1, 2, 3, 5, 6]
        required: true
        default: 1
        description:  User can choose any of the given model for training. System will override the existing model for selected model_id, action_id and protection_goods_id. Neural Network(scikit learn)=1, RANDOM FOREST(scikit learn)=2, LINEAR REGRESSION(scikit learn)=3, DEEP NEURAL NETWORK(Keras+tensorflow)=5, Decision Tree(scikit learn)=6
      - name: action_id
        in: formData
        type: integer
        required: true
        description: Value from AMUCAD application. Possible values corrosion=2, explosion=1
        default: 1
      - name: protection_goods_id
        in: formData
        type: integer
        required: true
        description: Value from AMUCAD application. Possible values 1,2,3,4,5
        default: 2
      - name: user_id
        in: formData
        type: string
        required: true
        description: 'User id of logged in user in AMUCAD application should be given'
        default: 2
      - name: token
        in: formData
        type: string
        required: true
        description:  'JSON Web Token, should be generated from login API using email and password'
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
    user_id = request.form.get('user_id')

    from app.structure import machine_learning as starter
    accuracy = 0
    try:
        obj = starter.MachineLearning(model_type, action_id, protection_goods_id, user_id)
        res = obj.accuracy_finder()
        status = res['status']
        ret = res['message']

        if status == 200:
            accuracy = res['accuracy']

    except Exception as e:
        status = 500
        ret = str(e)

    message = {
        'status': status,
        'accuracy': accuracy,
        'data': {
            'message': ret
        },
    }
    resp = jsonify(message)
    resp.status_code = status
    return resp


@app.route('/dss/evaluation', methods=['POST'])
@token_required
def dss_evaluation():
    """Endpoint for the assessment of the finding/findings.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: Input
        in: body
        type: string
        required: true
        description:  ' Array of findings; USER can also input multiple findings. For Data array, please follow the same sequence of features in which model was trained i.e [{
        "model_id":2,
        "protection_good_id": 2,
        "action_id": 1,
        "data": [{ "user_id":2, "model_id":6, "protection_good_id": 5, "action_id": 1, "data": [{"RP_shipping_all_2019": 23, "RP_shipping_cargo_2019": 11, "RP_shipping_container_2016": 0, "RP_shipping_fishing_2016": 1, "RP_shipping_other_2019": 0, "RP_shipping_passenger_2019": 0, "RP_shipping_rorocargo_2016": 1, "RP_shipping_service_2016": 1, "RP_shipping_tanker_2019": 6, "RP_seabed_slope": 0.279508501291275, "RP_depth": -123.440002441406, "OP_corrosion_level": null, "OP_sediment_cover": null, "OP_bio_cover": null, "OP_tnt_equivalent": 50, "CHARGE_113": 0, "CHARGE_114": 0, "CHARGE_115": 0, "CHARGE_117": 0, "CHARGE_118": 0, "CHARGE_119": 0, "CHARGE_120": 0, "CHARGE_121": 0, "CHARGE_124": 0, "CHARGE_125": 0, "CHARGE_126": 0, "CHARGE_127": 0, "CHARGE_129": 0, "CHARGE_130": 0, "CHARGE_131": 0, "CHARGE_132": 0, "CHARGE_135": 0, "CHARGE_136": 0, "CHARGE_137": 0, "CHARGE_138": 0, "CHARGE_139": 0, "CHARGE_140": 0, "CHARGE_142": 0, "CHARGE_143": 0, "CHARGE_144": 0, "CHARGE_145": 0, "CHARGE_146": 0, "CHARGE_147": 0, "CHARGE_148": 0, "CHARGE_154": 1, "CHARGE_155": 0, "CHARGE_156": 0, "CHARGE_157": 0, "CHARGE_158": 0, "CHARGE_159": 0, "CHARGE_160": 0, "CHARGE_161": 0, "CHARGE_162": 0, "CHARGE_163": 0, "CHARGE_164": 0, "CHARGE_165": 0, "CHARGE_166": 0, "CHARGE_167": 0, "CHARGE_168": 0, "CHARGE_169": 0, "CHARGE_170": 0, "CHARGE_171": 0, "CHARGE_172": 0, "CHARGE_173": 0, "CHARGE_174": 0, "CHARGE_175": 0}]}]'
      - name: token
        in: query
        type: string
        required: true
        description:  'JSON Web Token, should be generated from login API using email and password'
    responses:
      200:
        description: The response body contains the array of assessment objects for given data, action_id and protection_goods_id.
    """
    try:
        content = request.get_json(silent=True)
        from app.structure import machine_learning as starter

        results = []
        # check design pattern Chain Of Responsibility
        if content is not None:
            counter = 1
            for d in content:
                status = 400
                single_result = {}
                if 'model_id' not in d or 'action_id' not in d or 'protection_good_id' not in d or 'user_id' not in d:
                    single_result[
                        'assessment_' + str(counter)] = "Action ID, Model ID and Protection Good ID must be provided."
                    single_result['status'] = status
                else:
                    if 0 < d['model_id'] < 7:
                        obj = starter.MachineLearning(d['model_id'], d['action_id'], d['protection_good_id'],
                                                      d['user_id'])
                        single_result['model_id'] = d['model_id']
                        single_result['protection_good_id'] = d['protection_good_id']
                        single_result['action_id'] = d['action_id']
                        sample = []
                        for key in d['data'][0]:
                            if d['data'][0][key] is None:
                                sample.append(0)
                            else:
                                sample.append(d['data'][0][key])

                        obj.set_test_data(sample)
                        p_r = obj.testing()
                        if p_r is not None:
                            status = 200
                            single_result['assessment_response'] = p_r
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
        results.append({'message': str(e)})

    resp = jsonify(results)
    resp.status_code = status
    return resp


@app.route('/dss/logs', methods=['GET'])
@token_required
def dss_logs():
    """Endpoint to get history of actions for training by users.
    This is using docstrings for specifications.
    ---
    parameters:
      - name: token
        in: query
        type: string
        required: true
        description:  'JSON Web Token, should be generated from login API using email and password'

    responses:
      200:
        description: The response body contains the history of models training with  user_id.
    """
    from app.commons.mongo_connector import MongoConnector
    collection_training = MongoConnector.get_logsdb()
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
        description:  'Contact TUC for the Email and password'

      - name: password
        in: formData
        type: string
        format: password
        required: true
        description:  ''
    responses:
      200:
        description: A JSON object containing token to access further endpoints
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
            {'user': request.form.get('email'), 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=120)},
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
