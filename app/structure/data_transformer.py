import pandas as pd

from app import app
import os
import json
import requests
import hashlib
import base64
from requests_hawk import HawkAuth


def egeos_authentication_api():
    base_url = app.config['EGEOS']['base_url']
    r_login_request = requests.post(base_url + '/auth/login_request',
                                    data={'username': app.config['EGEOS']['user_name']})
    data = r_login_request.json()
    if "code" not in data or data["code"] != 401005:

        challenge = data['challenge']
        login_id = data['login_id']
        salt = data['salt']
        password = app.config['EGEOS']['password']

        salt = base64.b64decode(salt)
        challenge = base64.b64decode(challenge)
        concated = salt + password.encode('utf-8')
        currentHash = concated
        i = 0
        while i < 100000:
            hash_object = hashlib.sha256()
            hash_object.update(currentHash)
            currentHash = hash_object.digest()
            i += 1
        result = salt + currentHash
        digest1 = "digest1:" + str(base64.b64encode(result), 'utf-8')

        currentHash = salt + challenge + currentHash
        i = 0
        while i < 5:
            hash_object2 = hashlib.sha256()
            hash_object2.update(currentHash)
            currentHash = hash_object2.digest()
            i += 1

        challenge_response = str(base64.b64encode(currentHash), 'utf-8')
        headers = {"Accept-Language": "en-US,en;q=0.9,ur;q=0.8", "language": "eng"}
        challenge = str(base64.b64encode(challenge), 'utf-8')
        r_login = requests.post(base_url + '/auth/login',
                                data={'login_id': login_id, 'challenge': challenge,
                                      'challenge_response': challenge_response}, headers=headers)
        return r_login.json()
    else:
        return data


class Amucad:
    def __init__(self):
        self.__total_assessments = [
            'Explosion_Fisheries',
            'Explosion_Flora',
            'Explosion_Divers',
            'Explosion_Tourism',
            'Explosion_Fisheries',

            'Corrosion_Shipping',
            'Corrosion_Flora',
            'Corrosion_Divers',
            'Corrosion_Tourism',
            'Corrosion_Fisheries'
        ]
        self.__data = None

    def data_parser(self):
        data = {}
        for key in self.__data:
            data[key] = {}
            data[key]['id'] = key
            data[key]['confidence_level'] = self.__data[key]['confidence_level']
            data[key]['coordinates_0'] = self.__data[key]['finding']['geom']['coordinates'][0]
            data[key]['coordinates_1'] = self.__data[key]['finding']['geom']['coordinates'][1]
            self.extract_ammunitions(key, data)
            self.extract_traffic_intensity(key, data)
            self.extract_physical_features(key, data)
            self.extract_biodiversity(key, data)
            self.extract_fisheries(key, data)
            self.extract_bathymetry(key, data)
            self.extract_assessments(key, data)
        return data

    def transform_objects_to_csv(self):
        self.__data = self.load_data()
        # Transforming data into csv format
        self.export_data_to_csv(self.data_parser())

    def extract_assessments(self, key, data):
        for assessment in self.__data[key]['assessmentsAverage']:
            protection_good_name = assessment['name']
            for action in assessment['actions']:
                action_name = action['name']
                data[key][action_name + '_' + protection_good_name] = action['averageValue']

        for item in self.__total_assessments:
            if item not in data[key]:
                data[key][item] = None

    def extract_ammunitions(self, key, data):
        data[key]['ammunition_type_id'] = self.__data[key]['ammunitions'][0]['ammunition_types_id']
        data[key]['ammunition_type_name'] = self.__data[key]['ammunitions'][0]['ammunition_type']['name']

        # we should have names of following two options, id does not make sense
        data[key]['ammunition_categories_id'] = self.__data[key]['ammunitions'][0]['ammunition_type'][
            'ammunition_categories_id']
        data[key]['ammunition_sub_categories_id'] = self.__data[key]['ammunitions'][0]['ammunition_type'][
            'ammunition_sub_categories_id']

        try:
            data[key]['corrosion_level'] = \
                self.__data[key]['ammunitions'][0]['object_parameters'][0]['parameters_values']['value']
        except IndexError:
            data[key]['corrosion_level'] = None

        try:
            data[key]['sediment_cover'] = \
                self.__data[key]['ammunitions'][0]['object_parameters'][1]['parameters_values']['value']
        except IndexError:
            data[key]['sediment_cover'] = None

        try:
            data[key]['bio_cover'] = self.__data[key]['ammunitions'][0]['object_parameters'][2]['parameters_values'][
                'value']
        except IndexError:
            data[key]['bio_cover'] = None

    def extract_traffic_intensity(self, key, data):
        entity_name = self.__data[key]['environment']['regional_parameters'][0]['internal_name']
        for ti in self.__data[key]['environment']['regional_parameters'][0]['classifications']:
            data[key][entity_name + '_' + ti['data'] + '_value'] = ti['sources'][0]['values'][0]['value']
            data[key][entity_name + '_' + ti['data'] + '_value_classification_0'] = \
                ti['sources'][0]['values'][0]['classification'][0]
            data[key][entity_name + '_' + ti['data'] + '_value_classification_1'] = \
                ti['sources'][0]['values'][0]['classification'][1]

    def extract_physical_features(self, key, data):
        entity_name = self.__data[key]['environment']['regional_parameters'][1]['internal_name']

        for pf in self.__data[key]['environment']['regional_parameters'][1]['classifications']:

            if pf['data'] == "current_velocity" or pf['data'] == "salinity" or pf['data'] == "temperature":

                data[key][entity_name + '_' + pf['data'] + '_std'] = pf['sources'][0]['values'][0]['std']
                data[key][entity_name + '_' + pf['data'] + '_std_classification_0'] = \
                    pf['sources'][0]['values'][0]['classification'][0]
                data[key][entity_name + '_' + pf['data'] + '_std_classification_1'] = \
                    pf['sources'][0]['values'][0]['classification'][1]

                data[key][entity_name + '_' + pf['data'] + '_mean'] = pf['sources'][0]['values'][1]['mean']
                data[key][entity_name + '_' + pf['data'] + '_mean_classification_0'] = \
                    pf['sources'][0]['values'][1]['classification'][0]
                data[key][entity_name + '_' + pf['data'] + '_mean_classification_1'] = \
                    pf['sources'][0]['values'][1]['classification'][1]

            elif pf['data'] == "anoxic_level_probabilities" or pf['data'] == "oxygen_level_probabilities" or pf[
                'data'] == "seabed_slope":

                data[key][entity_name + '_' + pf['data'] + '_value'] = pf['sources'][0]['values'][0]['value']
                data[key][entity_name + '_' + pf['data'] + '_value_classification_0'] = \
                    pf['sources'][0]['values'][0]['classification'][0]
                data[key][entity_name + '_' + pf['data'] + '_value_classification_1'] = \
                    pf['sources'][0]['values'][0]['classification'][1]

    def extract_biodiversity(self, key, data):
        entity_name = self.__data[key]['environment']['regional_parameters'][2]['internal_name']
        for bd in self.__data[key]['environment']['regional_parameters'][2]['classifications']:
            if bd['data'] == "harbour_porpoises":
                data[key][entity_name + '_' + bd['data'] + '_value'] = bd['sources'][0]['values'][0]['value']
                data[key][entity_name + '_' + bd['data'] + '_value_classification_0'] = \
                    bd['sources'][0]['values'][0]['classification'][0]
                data[key][entity_name + '_' + bd['data'] + '_value_classification_1'] = \
                    bd['sources'][0]['values'][0]['classification'][1]
            else:
                data[key][entity_name + '_' + bd['data'] + '_bqr'] = bd['sources'][0]['values'][0]['bqr']
                data[key][entity_name + '_' + bd['data'] + '_bqr_classification_0'] = \
                    bd['sources'][0]['values'][0]['classification'][0]
                data[key][entity_name + '_' + bd['data'] + '_bqr_classification_1'] = \
                    bd['sources'][0]['values'][0]['classification'][1]

    def extract_fisheries(self, key, data):
        entity_name = self.__data[key]['environment']['regional_parameters'][3]['internal_name']
        for fs in self.__data[key]['environment']['regional_parameters'][3]['classifications']:
            data[key][entity_name + '_' + fs['data'] + '_value'] = fs['sources'][0]['values'][0]['value']
            data[key][entity_name + '_' + fs['data'] + '_value_classification_0'] = \
                fs['sources'][0]['values'][0]['classification'][0]
            data[key][entity_name + '_' + fs['data'] + '_value_classification_1'] = \
                fs['sources'][0]['values'][0]['classification'][1]

    def extract_bathymetry(self, key, data):
        entity_name = self.__data[key]['environment']['regional_parameters'][4]['internal_name']
        for fs in self.__data[key]['environment']['regional_parameters'][4]['classifications']:
            data[key][entity_name + '_' + fs['data'] + '_value'] = fs['sources'][0]['values'][0]['value']
            data[key][entity_name + '_' + fs['data'] + '_value_classification_0'] = \
                fs['sources'][0]['values'][0]['classification'][0]
            data[key][entity_name + '_' + fs['data'] + '_value_classification_1'] = \
                fs['sources'][0]['values'][0]['classification'][1]

    def load_data(self):

        # Loading data from file

        file = os.path.join(app.config['STORAGE_DIRECTORY'], "objects_regional_params.txt")
        if app.config['CACHE_API'] == 1 and os.path.exists(file):

            with open(file) as json_file:
                finding_objects = json.load(json_file)
        else:


            res = egeos_authentication_api()
            access_token = res['session_id']
            key = res['key']
            headers = {"language": "eng", "access_token": access_token}
            hawk_auth = HawkAuth(id=access_token, key=key, algorithm='sha256')

            data = requests.get("http://www.amucad.org/api/daimon/finding_objects?finding_object_types_id=2",
                                auth=hawk_auth, headers=headers)
            loaded_data = data.json()
            finding_objects = {}
            for munition in loaded_data['data']:
                data = requests.get("http://www.amucad.org/api/daimon/finding_objects/" + munition[
                    'id'] + "?$with_regional_params=true",
                                    auth=hawk_auth, headers=headers).json()
                finding_objects[data['id']] = data
            if app.config['CACHE_API'] == 1:
                with open(file, 'w') as outfile:
                    json.dump(finding_objects, outfile)

        return finding_objects

    def export_data_to_csv(self, data):

        directory = app.config['STORAGE_DIRECTORY']
        filename = "amucad_dataset.csv"
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        data_frame = pd.DataFrame.from_dict(data, orient='index')
        data_frame.to_csv(file_path, index=False)

    def get_object_detail(self, object_id):

        res = egeos_authentication_api()
        access_token = res['session_id']
        key = res['key']
        headers = {"language": "eng", "access_token": access_token}
        hawk_auth = HawkAuth(id=access_token, key=key, algorithm='sha256')

        response = requests.get(
            "http://www.amucad.org/api/daimon/finding_objects/" + str(object_id) + "?$with_regional_params=true",
            auth=hawk_auth, headers=headers).json()

        if 'id' in response:
            self.__data = {object_id: response}
            return self.data_parser()[object_id]
        else:
            return None

    def amucad_generic_api(self, obj):

        res = egeos_authentication_api()
        access_token = res['session_id']
        key = res['key']
        headers = {"language": "eng", "access_token": access_token}
        hawk_auth = HawkAuth(id=access_token, key=key, algorithm='sha256')
        page_size = 50
        page_index = 0
        total_pages = 1

        input_variables = {}
        output_variables = {}
        assessments_avg = {}
        i = 0
        # obj.action_id, obj.protection_goods_id
        fileName = str(obj.action_id) + "_" + str(obj.protection_goods_id)+ "_dynamic_data.txt"
        file = os.path.join(app.config['STORAGE_DIRECTORY'], fileName)
        if app.config['CACHE_API'] == 1 and os.path.exists(file):

            with open(file) as json_file:
                api_response = json.load(json_file)

            output_variables = api_response['output_variables']
            input_variables = api_response['input_variables']
            for d in api_response['data']:
                assessments_avg[i] = {}
                assessments_avg[i] = d
                i = i + 1

        else:

            while (page_index + 1) <= total_pages:
                url = 'http://amucad.org/api/decision_support_system/dss_training/' \
                      'assessed_finding_objects?actions_id={}&protection_goods_id={}&' \
                      '$page_size={}&$page_index={}'.format(obj.action_id, obj.protection_goods_id, page_size,
                                                            page_index)
                api_response = requests.get(url,
                                            auth=hawk_auth, headers=headers).json()

                output_variables = api_response['output_variables']
                input_variables = api_response['input_variables']
                total_pages = api_response['total_pages']
                page_index = page_index + 1
                for d in api_response['data']:
                    assessments_avg[i] = {}
                    assessments_avg[i] = d
                    i = i + 1

                if app.config['CACHE_API'] == 1:
                    with open(file, 'w') as outfile:
                        json.dump(api_response, outfile)

        return pd.DataFrame.from_dict(assessments_avg, orient='index'), input_variables, output_variables
