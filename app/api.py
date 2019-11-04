from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss

# from marshmallow import Schema, fields
from flask import abort

from marshmallow import Schema, fields, validate, ValidationError



class CreateDSSInputSchema(Schema):

    model_id = fields.Int(required=True,validate=validate.Range(min=1, max=4))
    training = fields.Int(required=True,validate=validate.Range(min=0, max=1))
    testing  = fields.Int(required=True,validate=validate.Range(min=0, max=1))





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
    testing  = 0 
    if request.form.get('model_id'):
        model_id = int(request.form.get('model_id'))
    if request.form.get('training'):
        training = int(request.form.get('training'))
    if request.form.get('testing'):
        testing  = int(request.form.get('testing'))


    if model_id:
        if(model_id == 1):
            model = dss.NeuralNetwork(app.config['NEURAL_NETWORK_MODEL'])
        elif(model_id == 2):
            model = dss.RandomForest(app.config['RANDOM_FOREST_CLASSIFIER_MODEL'])
        elif(model_id == 3):
            model = dss.LinearRegressionM(app.config['LINEAR_REGRESSION_MODEL'])
        elif(model_id == 4):
            model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])
        
        
        data = None
    
        if( training == 1 ):
            
            model.data_intialization()
            model.data_preprocessing()
            modelObject = model.training()

            if(modelObject.get_trained_model() != None):

                model.save_model( modelObject.get_trained_model() )
                data = {
                    'model_id' :    model_id,
                    'model_name'  : modelObject.get_name(),
                    'status':  200,
                    'message': 'Trained Successully',
                    'results': [],
                    'accuracy': modelObject.get_accuracy()
                }
            else:

                data = {
                    'model_id' :    model_id,
                    'model_name'  : modelObject.get_name(),
                    'status':  500,
                    'message': 'Model is empty',
                    'results': [],
                    'accuracy': None
                }


        if( testing == 1):
            model_reseponse =  model.testing()
            data = {
                'model_id' :    model_id,
                'model_name'  : model_name,
                'status':  200,
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
    return  'awais'
    
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
def bad_request(error=None):
    message = {
            'status': 404,
            'message': error,
    }
    resp = jsonify(message)
    resp.status_code = 404
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


       

