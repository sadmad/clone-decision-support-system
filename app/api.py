from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss
from flask_restful import reqparse




@app.route('/dss', methods=['GET'])
def dss_main():

    model = None
    model_name = training = testing = None
    

    if request.args.get('model_id'):
        model_id = int(request.args.get('model_id'))
    if request.args.get('training'):
        training = int(request.args.get('training'))
    if request.args.get('testing'):
        testing  = int(request.args.get('testing'))

    if model_id:
        if(model_id == 1):
            model = dss.NeuralNetwork(app.config['NEURAL_NETWORK_MODEL'])
            model_name = 'NEURAL_NETWORK_MODEL'
        elif(model_id == 2):
            model = dss.RandomForest(app.config['RANDOM_FOREST_CLASSIFIER_MODEL'])
            model_name = 'RANDOM_FOREST_CLASSIFIER_MODEL'
        elif(model_id == 3):
            model = dss.LinearRegressionM(app.config['LINEAR_REGRESSION_MODEL'])
            model_name = 'LINEAR_REGRESSION_MODEL'
        elif(model_id == 4):
            model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])
            model_name = 'LOGISTIC_REGRESSION_MODEL'

        
        data = None
        if training:
            if( training == 1 ):
                model.data_intialization()
                model.data_preprocessing()
                trained_model = model.training()
                model.save_model( trained_model )
                data = {
                    'model_id' :    model_id,
                    'model_name'  : model_name,
                    'status':  200,
                    'message': 'Trained Successully',
                    'results': model_reseponse
                }


        if testing:
            if( testing == 1):
                model_reseponse =  model.testing()
                data = {
                    'model_id' :    model_id,
                    'model_name'  : model_name,
                    'status':  200,
                    'message': 'Tested successully',
                    'results': model_reseponse
                }
                print(data)

        resp = jsonify(data)
        resp.status_code = 200
        return resp

    else:
        return not_found()




    
    
@app.errorhandler(404)
def not_found(error=None):
    message = {
            'status': 404,
            'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

@app.errorhandler(400)
def not_found(error=None):
    message = {
            'status': 400,
            'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 400
    return resp
    

@app.errorhandler(500)
def internal_error(error):
    message = {
            'status': 500,
            'message': 'Some Internal Server Error: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 500
    return resp


       
