from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss

# from marshmallow import Schema, fields
from flask import abort

from marshmallow import Schema, fields, validate, ValidationError


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



# import pycurl
# from io import BytesIO
# from urllib.parse import urlencode
# @app.route('/amucad/api/authentication/', methods=['GET'])
# def egeos_authentication_api():
#     crl = pycurl.Curl()
#     crl.setopt(crl.URL, 'https://mdb.in.tu-clausthal.de/api/auth/login')
#     data = {'email': 'dst11.admin@tu-clausthal.de','password': 'qJ2vLNgrA7EP8KKw'}
#     pf = urlencode(data)

#     # Sets request method to POST,
#     # Content-Type header to application/x-www-form-urlencoded
#     # and data to send in request body.
#     crl.setopt(crl.POSTFIELDS, pf)
#     crl.perform()
#     crl.close()
#     status_code = crl.getinfo(pycurl.RESPONSE_CODE)
#     return status_code

import requests
import sys
from hashlib import sha256
import binascii, os
import base64
from Crypto.Hash import SHA256
# from Crypto.Hash import SHA256
@app.route('/amucad/api/authentication/', methods=['GET'])
def egeos_authentication_api():
    

    # r = requests.post('https://mdb.in.tu-clausthal.de/api/v1/auth/login/', data = {'email': 'dst11.admin@tu-clausthal.de','password': 'qJ2vLNgrA7EP8KKw'})
    # return r.json()


    #r_login_request = requests.post('https://www.amucad.org/auth/login_request', data = {'username': 'AMushtaq'})
 
    #return r_login_request.json()

    # challenge   = r_login_request.json()['challenge']
    # login_id    = r_login_request.json()['login_id']
    #salt        = r_login_request.json()['salt']
    password    = "lK98hgr&h"


    challenge   = "1nVqYpZqjj0wweA08ExSra8NOIwW/Yer+00xx3Vjbqk="
    # login_id    = "542"
    salt        = "j3zpl6wHbivcG2phZsw8Kw=="
    # password    = "lK98hgr&h"
   
    passwordBuffer = bytes(password.encode("utf-8").hex(), "utf-8")

    saltBuffer     = bytes(salt.encode("utf-8").hex(), "utf-8")

    concated = saltBuffer+passwordBuffer
    #concated = b"".join([saltBuffer, passwordBuffer])
    #return bytearray.fromhex(str(concated, 'utf-8')).decode()
        
    currentHash = concated
    #currentHash = bytearray.fromhex(str(concated, 'utf-8')).decode()
    #return currentHash


    i = 0
    while i < 100000:
        # H = sha256()
        # H.update(currentHash)
        # currentHash = H.digest()


        hash_object = SHA256.new()
        hash_object.update(currentHash)
        currentHash = hash_object.digest()
        i += 1

    #return currentHash
    #return str(base64.b64encode(currentHash), 'utf-8')
    
    result = b"".join([saltBuffer, currentHash])

    #return 
    # for character in result:
    #     print(character, character.encode('utf-8').hex())
    digest1        =  "digest1:" + str(base64.b64encode(result), 'utf-8')
    pureDigestStr  = digest1.replace("digest1:", "")
    buf            = base64.b64encode(pureDigestStr.encode())
    salt           = buf[:16]
    hashedPassword = buf[16:-1]
    currentHash    = salt + bytes(challenge.encode("utf-8").hex(), "utf-8") + hashedPassword

    i = 0
    while i < 5:
        # H = sha256()
        # H.update(currentHash)
        # currentHash = H.digest()
        hash_object = SHA256.new()
        hash_object.update(currentHash)
        currentHash = hash_object.digest()
        i += 1

    return currentHash
    #r_login = requests.post('https://www.amucad.org/auth/login', data = {'login_id': 'AMushtaq','challenge': 'AMushtaq','salt': 'AMushtaq'})


from app import errors