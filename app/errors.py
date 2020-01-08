from app import app
from flask import request,jsonify


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 404,
        'message': error,
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

@app.errorhandler(404)
def not_found(error=None):
    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 404
    return resp

@app.errorhandler(405)
def not_found(error=None):
    message = {
        'status': 405,
        'message': 'Method type not allowed: ' + request.url,
    }
    resp = jsonify(message)
    resp.status_code = 405
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


