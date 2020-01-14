from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss

# from marshmallow import Schema, fields
from flask import abort

from marshmallow import Schema, fields, validate, ValidationError
import requests
#import sys
import hashlib
#import binascii, os
import base64


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
            model = dss.NeuralNetwork(app.config['NEURAL_NETWORK_MODEL'])
        elif (model_id == 2):
            model = dss.RandomForest(app.config['RANDOM_FOREST_CLASSIFIER_MODEL'])
        elif (model_id == 3):
            model = dss.LinearRegressionM(app.config['LINEAR_REGRESSION_MODEL'])
        elif (model_id == 4):
            model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])

        data = None

        if training == 1:

            model.data_intialization()
            model.data_preprocessing()
            modelObject = model.training()

            if modelObject.get_trained_model() is not None:

                model.save_model(modelObject.get_trained_model())
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
            model_reseponse = model.testing()
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




@app.route('/amucad/api/authentication/', methods=['GET'])
def egeos_authentication_api():


    base_url = app.config['EGEOS']['base_url']
    r_login_request = requests.post(base_url+'/auth/login_request', data = {'username':app.config['EGEOS']['user_name']})
    data = r_login_request.json()
    if "code" not in data or data["code"] != 401005:
        print(r_login_request.json())
        challenge   = data['challenge']
        login_id    = data['login_id']
        salt        = data['salt']
        password    = app.config['EGEOS']['password']

        salt = base64.b64decode(salt)
        challenge = base64.b64decode(challenge)
        concated    = salt + password.encode('utf-8')
        currentHash = concated
        i = 0
        while i < 100000:
            hash_object = hashlib.sha256()
            hash_object.update(currentHash)
            currentHash = hash_object.digest()
            i += 1
        result = salt + currentHash
        digest1        =  "digest1:" + str(base64.b64encode(result), 'utf-8')

        currentHash    = salt + challenge + currentHash
        i = 0
        while i < 5:
            hash_object2 = hashlib.sha256()
            hash_object2.update(currentHash)
            currentHash = hash_object2.digest()
            i += 1

        challenge_response = str(base64.b64encode(currentHash), 'utf-8')
        headers = {"Accept-Language": "en-US,en;q=0.9,ur;q=0.8","language": "eng"}
        challenge = str(base64.b64encode(challenge),  'utf-8')
        r_login = requests.post('https://www.amucad.org/auth/login', data = {'login_id': login_id,'challenge': challenge,'challenge_response': challenge_response},  headers=headers)
        return r_login.json()
    else:
        return data


from requests_hawk import HawkAuth
@app.route('/amucad/api/find_amucad_objects/', methods=['GET'])
def find_amucad_objects():
    access_token = 'a0098c75a697461099664f6068f93c56'
    key = 'dca17c503ca565d8ba2baba1a2623dfd'
    headers = {"language": "eng","access_token":access_token}
    hawk_auth = HawkAuth(id=access_token, key=key, algorithm ='sha256')
    data = requests.get("http://www.amucad.org/api/daimon/finding_objects", auth=hawk_auth,  headers=headers)
    return data.json()













from app import errors