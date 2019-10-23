from app import app
import os
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
#import pickle
from flask import request, render_template
import joblib
from app import scale
import pandas as pd
from flask_basicauth import BasicAuth
from app.structure import dss

books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]

features = [
    "Cultivator", 
    "Alchol", 
    "Malic_Acid", 
    "Ash", 
    "Alcalinity_of_Ash", 
    "Magnesium", 
    "Total_phenols", 
    "Falvanoids", 
    "Nonflavanoid_phenols", 
    "Proanthocyanins", 
    "Color_intensity", 
    "Hue", 
    "OD280", 
    "Proline"
]


@app.route('/api/v1/resources/books/all', methods=['GET'])
def api_all():
    return jsonify(books)


@app.route('/api/v1/resources/books', methods=['GET'])
def api_id():
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []
    for book in books:
        if book['id'] == id:
            results.append(book)

    return jsonify(results)

# a simple page that says hello
@app.route('/neural-network/train-model')
def neural_network_train_model():
    #https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
    train_data = pd.read_csv('wine_data_train.csv', names = features)
    
    X_train = train_data.drop('Cultivator',axis=1)
    Y_train = train_data['Cultivator']
    

    X_train = scale.Scale.StandardScaler( X_train )



    model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
    model.fit(X_train,Y_train)


    file_path = os.path.join(app_path,app.config['NEURAL_NETWORK_MODEL'] )
    if os.path.exists(file_path):
        os.remove(file_path)


    # save the model to disk
    #https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

    #pickle.dump(model, open(app.config['NEURAL_NETWORK_MODEL'], 'wb'))
    joblib.dump(model, file_path)
    return jsonify([{
            'status':200,
            'message':'Neural Network Model is saved successfully'
        }])



# Test Neural Network on Test File
@app.route('/neural-network/test-model')
def neural_network_test_model():

    test_data  = pd.read_csv('wine_data_test.csv',  names = features)
    
    X_test  = test_data.drop('Cultivator',axis=1)
    Y_test  = test_data['Cultivator']



    X_test = scale.Scale.LoadScalerTest( X_test )



    # load the model from disk
    file_path = os.path.join(app_path,app.config['NEURAL_NETWORK_MODEL'] )
    #loaded_model = pickle.load(open(file_path, 'rb'))
    loaded_model = joblib.load(file_path)
    score_result = loaded_model.score(X_test, Y_test)
    predictions  = loaded_model.predict(X_test)
    print(predictions)
    # print(confusion_matrix(Y_test,predictions))
    # print(classification_report(Y_test,predictions))

    return jsonify([{
        'status':200,
        'message':'Test Obervations are predicted by Neural Network Trained Model.',
        'predictions' : pd.Series(predictions).to_json(orient='values')
    }])



# Test Neural Network on Test File
@app.route('/dss', methods=['GET'])
def dss_main( ):
    
    model = dss.NeuralNetwork('NEURAL_NETWORK_MODEL')

    model.data_intialization()
    model.data_preprocessing()
    trained_model = model.model()
    model.save_model( trained_model )

    return model.testing()