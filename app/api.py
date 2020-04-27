import math

from app import app
import os
from flask import Flask, request, jsonify, render_template

from app.input_schema import FDIInputSchema, CFInputSchema, TrainingAPISchema, MunitionInputSchema
from app.structure import dss
# from app.structure import data_transformer as DT
# from marshmallow import Schema, fields
from flask import abort
from app import errors

from marshmallow import Schema, fields, validate, ValidationError
from app.structure import model as md, data_transformer as amc
import redis
import json


class CreateDSSInputSchema(Schema):
    model_id = fields.Int(required=True, validate=validate.Range(min=1, max=4))
    training = fields.Int(required=True, validate=validate.Range(min=0, max=1))
    testing = fields.Int(required=True, validate=validate.Range(min=0, max=1))


@app.route('/cross-validation/k-fold', methods=['GET'])
def cross_validation_kfold_main():
    model = dss.LogisticRegressionM(app.config['LOGISTIC_REGRESSION_MODEL'])
    accuracy = model.kfold()
    return 'awais'


@app.route('/amucad/api/data_transformation/', methods=['GET'])
def find_amucad_objects():
    obj = amc.Amucad()
    obj.transform_objects_to_csv()
    return 'true'


@app.route('/dss/training', methods=['POST'])
def fish_training():
    """
    fish_training(arg1, arg2=None) --> does training of models

    Parameters:
    arg1: abc.
    arg2: edf.
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
        mdObject = md.ExplosionFisheriesAssessment(model_type)
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

    elif data['assessment_id'] == 3:
        validation = MunitionInputSchema().validate(request.form)
        if validation:
            message = {
                'status': 422,
                'message': str(validation),
            }
            resp = jsonify(message)
            resp.status_code = 422
            return resp
        data['model_id'] = int(request.form.get('model_id'))
        data['object_id'] = int(request.form.get('object_id'))
        ob = amc.Amucad()
        object_id = data['object_id']
        rs = ob.get_object_detail(object_id)
        mdObject = md.ExplosionFisheriesAssessment(model_type=data['model_id'])

        for key in rs:
            if rs[key] is None:
                rs[key] = 0
        col_32 = col_33 = col_34 = col_35 = col_36 = col_37 = col_38 = col_39 = col_40 = 0
        if rs['ammunition_type_id'] == 137:
            col_33 = 1
        if rs['ammunition_type_id'] == 138:
            col_34 = 1
        if rs['ammunition_type_id'] == 139:
            col_35 = 1
        if rs['ammunition_type_id'] == 140:
            col_36 = 1
        if rs['ammunition_type_id'] == 141:
            col_37 = 1
        if rs['ammunition_type_id'] == 85:
            col_32 = 1

        if rs['ammunition_categories_id'] == 1:
            col_38 = 1
        if rs['ammunition_sub_categories_id'] == 1:
            col_39 = 1
        if rs['ammunition_sub_categories_id'] == 2:
            col_40 = 1
        prediction = mdObject.predict_data([[
            rs['confidence_level'], rs['coordinates_0'], rs['coordinates_1'],
            rs['corrosion_level'], rs['sediment_cover'], rs['bio_cover'],
            rs['traffic_intensity_shipping_all_2016_value'], rs['traffic_intensity_shipping_cargo_2016_value'], rs['traffic_intensity_shipping_container_2016_value'],
            rs['traffic_intensity_shipping_fishing_2016_value'], rs['traffic_intensity_shipping_other_2016_value'], rs['traffic_intensity_shipping_passenger_2016_value'],
            rs['traffic_intensity_shipping_rorocargo_2016_value'], rs['traffic_intensity_shipping_service_2016_value'], rs['traffic_intensity_shipping_tanker_2016_value'],
            rs['physical_features_current_velocity_std'], rs['physical_features_current_velocity_mean'], rs['physical_features_anoxic_level_probabilities_value'],
            rs['physical_features_oxygen_level_probabilities_value'], rs['physical_features_seabed_slope_value'], rs['physical_features_salinity_std'], rs['physical_features_salinity_mean'],
            rs['physical_features_temperature_std'], rs['physical_features_temperature_mean'], rs['biodiversity_benthic_habitats_bqr'],
            rs['biodiversity_pelagic_habitats_bqr'], rs['biodiversity_integrated_fish_assessments_bqr'], rs['biodiversity_harbour_porpoises_value'], rs['fisheries_fisheries_bottom_trawl_value'],
            rs['fisheries_fisheries_surface_midwater_value'], rs['fisheries_coastal_and_stationary_value'], rs['bathymetry_depth_value'],
            col_32, col_33, col_34,
            col_35, col_36, col_37,
            col_38, col_39, col_40
        ]])
        print(rs)

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
