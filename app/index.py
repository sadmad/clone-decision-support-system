import requests
import sys
import hashlib
import binascii, os
import base64

def egeos_authentication_api():
    

    # r = requests.post('https://mdb.in.tu-clausthal.de/api/v1/auth/login/', data = {'email': 'dst11.admin@tu-clausthal.de','password': 'qJ2vLNgrA7EP8KKw'})
    # return r.json()
    r_login_request = requests.post('https://www.amucad.org/auth/login_request', data = {'username': 'Enter USername'})
    data = r_login_request.json()
    if "code" not in data or data["code"] != 401005:
        print(r_login_request.json())
        challenge   = data['challenge']
        login_id    = data['login_id']
        salt        = data['salt']

        password    = "Password"
        #challenge   = "s7lcwRFS/WiZA94LgXMxo8mPDX2EdOcoWFZwleaTbIE="
        #salt        = "j3zpl6wHbivcG2phZsw8Kw=="
       
        concated    = salt.encode('utf-8') + password.encode('utf-8')
        
        currentHash = concated
        i = 0
        while i < 100000:
            hash_object = hashlib.sha256()
            hash_object.update(currentHash)
            currentHash = hash_object.digest()
            i += 1

        result = salt.encode('utf-8') + currentHash
        digest1        =  "digest1:" + str(base64.b64encode(result), 'utf-8')

        #return digest1
        pureDigestStr  = digest1.replace("digest1:", "")
        buf            = base64.b64decode(pureDigestStr)
        
        salt           = buf[:16]
        hashedPassword = buf[16:]
        currentHash    = salt + challenge.encode('utf-8') + hashedPassword
        
        i = 0
        while i < 5:
            hash_object2 = hashlib.sha256()
            hash_object2.update(currentHash)
            currentHash = hash_object2.digest()
            i += 1

        #return currentHash
        challenge_response = str(base64.b64encode(currentHash), 'utf-8')
        #challenge_response = currentHash

        headers = {"Accept-Language": "en-US,en;q=0.9,ur;q=0.8","language": "eng"}
        print('login_id = '+ login_id)
        print('challenge = '+ challenge)
        print('challenge_response  Base64 String = '+ str(base64.b64encode(currentHash), 'utf-8'))
        r_login = requests.post('https://www.amucad.org/auth/login', data = {'login_id': login_id,'challenge': challenge,'challenge_response': challenge_response},  headers=headers)
        return r_login.json()
    else:
        return data


egeos_authentication_api()