from app import app
import os
from flask import Flask, request, jsonify, render_template
from app.structure import dss
# from app.structure import data_transformer as DT
# from marshmallow import Schema, fields
from flask import abort

from marshmallow import Schema, fields, validate, ValidationError
from app.structure import model as md


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


@app.route('/fish/training', methods=['GET'])
def fish_training():
    model_type = 1  # Random Forest

    # DSS model type is initialized
    mdObject = md.FdiAssessment(model_type)

    mdObject.start();

    return 'awais'


# https://github.com/marshmallow-code/marshmallow

def validate_sex(n):
    if n != 'm' and n != 'w' and n != 'n':
        raise ValidationError("Sex should be m,w or n")


class CreateRTInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))
    assessment_id = fields.Int(required=True, validate=validate.Range(min=1, max=10))

    station = fields.Int(required=True)
    year = fields.Int(required=True)
    month = fields.Int(required=True)
    day = fields.Int(required=True)
    group = fields.Str(required=True)
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
    data['group'] = request.form.get('group')
    data['sex'] = request.form.get('sex')
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
        mdObject = md.FdiAssessment(model_type=data['model_id'])
    print(data)

    return 'Waleed'


from app import errors
