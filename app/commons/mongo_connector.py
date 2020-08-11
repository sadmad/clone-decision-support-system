from pymongo import MongoClient
from app import app


class MongoConnector:
    @staticmethod
    def get_client():
        return MongoClient('mongodb://localhost:27017/',
                           username=app.config['MONGO_DB_USERNAME'],
                           password=app.config['MONGO_DB_PASSWORD'],
                           authSource=app.config['MONGO_DB_AUTHSOURCE'],
                           authMechanism='SCRAM-SHA-256')

    @staticmethod
    def get_logsdb():
        client = MongoConnector.get_client()
        return client[app.config['MONGO_DB_AUTHSOURCE']]['logs']
