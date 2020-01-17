from app import app
import os
import json
import requests
import hashlib
import base64
from requests_hawk import HawkAuth

class DataTransformer:

    def __init__( self ):
        self.__data = self.load_data()
        

    def get_data( self ):
        return self.__data

    def load_data( self ):

        if app.config['CACHE_API'] == 1 and os.path.exists( "api_data.txt" ):
            print(' Loading data from File ')
            with open('api_data.txt') as json_file:
                loaded_data = json.load(json_file)    

        else:

            print(' Loading data from API ')
            res = self.egeos_authentication_api()
            access_token = res['session_id']
            key = res['key']
            headers = {"language": "eng","access_token":access_token}
            hawk_auth = HawkAuth(id=access_token, key=key, algorithm ='sha256')
            data = requests.get("http://www.amucad.org/api/daimon/finding_objects", auth=hawk_auth,  headers=headers)
            
            loaded_data = data.json()

            if app.config['CACHE_API'] == 1:
                with open('api_data.txt', 'w') as outfile:
                    json.dump( loaded_data, outfile )

        return loaded_data

    def egeos_authentication_api( self ):

        base_url = app.config['EGEOS']['base_url']
        r_login_request = requests.post(base_url+'/auth/login_request', data = {'username':app.config['EGEOS']['user_name']})
        data = r_login_request.json()
        if "code" not in data or data["code"] != 401005:
            print(r_login_request.json())
            challenge   = data['challenge']
            login_id    = data['login_id']
            salt        = data['salt']
            password    = app.config['EGEOS']['password']

            salt = base64.b64decode(salt)
            challenge = base64.b64decode(challenge)
            concated    = salt + password.encode('utf-8')
            currentHash = concated
            i = 0
            while i < 100000:
                hash_object = hashlib.sha256()
                hash_object.update(currentHash)
                currentHash = hash_object.digest()
                i += 1
            result = salt + currentHash
            digest1        =  "digest1:" + str(base64.b64encode(result), 'utf-8')

            currentHash    = salt + challenge + currentHash
            i = 0
            while i < 5:
                hash_object2 = hashlib.sha256()
                hash_object2.update(currentHash)
                currentHash = hash_object2.digest()
                i += 1

            challenge_response = str(base64.b64encode(currentHash), 'utf-8')
            headers = {"Accept-Language": "en-US,en;q=0.9,ur;q=0.8","language": "eng"}
            challenge = str(base64.b64encode(challenge),  'utf-8')
            r_login = requests.post('https://www.amucad.org/auth/login', data = {'login_id': login_id,'challenge': challenge,'challenge_response': challenge_response},  headers=headers)
            return r_login.json()
        else:
            return data

    def export_data_to_csv( self ):
        return 'awais'

