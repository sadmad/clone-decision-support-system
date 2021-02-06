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

    def amucad_generic_api(self, obj):

        input_variables = {}
        output_variables = {}
        assessments_avg = {}
        i = 0
        # # obj.action_id, obj.protection_goods_id
        fileName = str(obj.action_id) + "_" + str(obj.protection_goods_id) + "_dynamic_data.txt"
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
            res = egeos_authentication_api()
            access_token = res['session_id']
            key = res['key']
            headers = {"language": "eng", "access_token": access_token}
            hawk_auth = HawkAuth(id=access_token, key=key, algorithm='sha256')
            page_size = 50
            page_index = 0
            total_pages = 1

            i = 0
            # obj.action_id, obj.protection_goods_id
            fileName = str(obj.action_id) + "_" + str(obj.protection_goods_id) + "_dynamic_data.txt"
            file = os.path.join(app.config['STORAGE_DIRECTORY'], fileName)

            while (page_index + 1) <= total_pages:
                url = 'http://www.amucad.org/api/decision_support_system/dss_training/' \
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
