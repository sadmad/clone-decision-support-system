from app import app
import os
from flask import Flask, request, jsonify

from flask import request, render_template
from app.structure import dss

@app.route('/dss', methods=['GET'])
def dss_main( ):
    
    
    
    #model = dss.NeuralNetwork(app.config['NEURAL_NETWORK_MODEL'])    
    #model = dss.RandomForest(app.config['RANDOM_FOREST_CLASSIFIER_MODEL'])
    #model = dss.LinearRegressionM(app.config['LINEAR_REGRESSION_MODEL'])

    model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])


    model.data_intialization()
    model.data_preprocessing()
    trained_model = model.training()
    model.save_model( trained_model )

    return model.testing()