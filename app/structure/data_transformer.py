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
        print(r_login_request.json())
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
        r_login = requests.post('https://www.amucad.org/auth/login',
                                data={'login_id': login_id, 'challenge': challenge,
                                      'challenge_response': challenge_response}, headers=headers)
        return r_login.json()
    else:
        return data


class DataTransformer:

    def __init__(self):
        self.__data = self.load_data()

        # Transforming data into csv format
        data = {}
        for key in self.__data:
            data[key] = {}
            data[key]['id'] = key
            self.extract_traffic_intensity(key, data)
            self.extract_physical_features(key, data)
            self.extract_biodiversity(key, data)
            self.extract_fisheries(key, data)
            self.extract_bathymetry(key, data)

        self.export_data_to_csv(data)
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

    def get_data(self):
        return self.__data

    def load_data(self):

        # Loading data from file
        if app.config['CACHE_API'] == 1 and os.path.exists("objects_regional_params.txt"):
            print(' Loading data from File ')
            with open('objects_regional_params.txt') as json_file:
                finding_objects = json.load(json_file)
        else:

            print(' Loading data from API ')
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
                with open('objects_regional_params.txt', 'w') as outfile:
                    json.dump(finding_objects, outfile)

        return finding_objects

    def export_data_to_csv(self, data):

        file_name = "amucad_dataset.csv"
        if os.path.exists(file_name):
            os.remove(file_name)

        data_frame = pd.DataFrame.from_dict(data, orient='index')
        data_frame.to_csv(r'amucad_dataset.csv', index=False)
        
