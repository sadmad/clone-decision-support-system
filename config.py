import os
from app import app


class Config:
    """
    Use this class to share any default attributes with any subsequent
    classes that inherit from Config.
    """
    DEBUG = False
    TESTING = False

    # Only required when using the session object
    # Generated with secrets.token_urlsafe(16)
    # You could also use os.urandom(16)
    SECRET_KEY = "11Fnw8U6DXrMFvbH9jCdZQ"

    NEURAL_NETWORK_MODEL = {
        'model': 'neural_network.sav',
        'scaler': 'neural_network_scaler.save'
    }

    RANDOM_FOREST_CLASSIFIER_MODEL = {
        'model': 'random_forest_classifier.sav',
        'scaler': 'random_forest_classifier_scaler.save'
    }

    LINEAR_REGRESSION_MODEL = {
        'model': 'linear_regression.sav',
        'scaler': 'linear_regression_scaler.save'
    }

    LOGISTIC_REGRESSION_MODEL = {
        'model': 'logistic_regression.sav',
        'scaler': 'logistic_regression_scaler.save'
    }
    DEEP_NEURAL_NETWORK_MODEL = {
        'model': 'deep_neural_network.sav',
        'scaler': 'deep_neural_network_scaler.save'
    }

    DECISIONTREE_REGRESSOR_MODEL = {
        'model': 'decision_tree.sav',
        'scaler': 'decision_tree.save'
    }

    MODELS = {

        'FDI_ASSESSMENT': 'fdi_assessment_',  # FISH
        'CF_ASSESSMENT': 'cf_assessment_',  # FISH
        'LHI_ASSESSMENT': 'lhi_assessment_',  # FISH
        'MUSCEL_CWA_ASSESSMENT': 'muscel_cwa_assessment_',  # FISH
        'LIVER_CWA_ASSESSMENT': 'liver_cwa_assessment_',  # FISH
        'ERY_ASSESSMENT': 'ery_assessment_',  # FISH
        'HB_ASSESSMENT': 'hb_assessment_',  # FISH
        'GLU_ASSESSMENT': 'glu_assessment_',  # FISH
        'HCT_ASSESSMENT': 'hct_assessment_',  # FISH
        'GILL_CWA_ASSESSMENT': 'gill_cwa_assessment_',  # FISH
        'EXPLOSION_FISHERIES_ASSESSMENT': 'explosion_fisheries_assessment',  # Munition
        'EXPLOSION_SHIPPING_ASSESSMENT': 'explosion_shipping_assessment'  # Munition
    }

    MODELS_ID_MAPPING = {
        0: 'FDI_ASSESSMENT',
        1: 'CF_ASSESSMENT',
        2: 'LHI_ASSESSMENT',
        3: 'MUSCEL_CWA_ASSESSMENT',
        4: 'LIVER_CWA_ASSESSMENT',
        5: 'ERY_ASSESSMENT',
        6: 'HB_ASSESSMENT',
        7: 'GLU_ASSESSMENT',
        8: 'HCT_ASSESSMENT',
        9: 'GILL_CWA_ASSESSMENT',
        10: 'EXPLOSION_FISHERIES_ASSESSMENT',  # Munition
        11: 'EXPLOSION_SHIPPING_ASSESSMENT',  # Munition
    }

    EGEOS = {
        'base_url': 'https://www.amucad.org',
        'user_name': 'AMushtaq',
        'password': 'lK98hgr&h'
    }

    # EGEOS = {
    #     'base_url':  'https://development-amucad.egeos.de',
    #     'user_name': 'AWais',
    #     'password':  'Kfy$482xx'
    # }

    SWAGGER_CONF = {
        "headers": [
        ],
        "specs": [
            {
                "endpoint": 'apispec_1',
                "route": '/apispec_1.json',
                "rule_filter": lambda rule: True,  # all in
                "model_filter": lambda tag: True,  # all in
            }
        ],
        "static_url_path": "/flasgger_static",
        # "static_folder": "static",  # must be set by user
        "swagger_ui": True,
        "specs_route": "/apidocs/",
        'title': 'DSS API Documentation'
    }

    SWAGGER_TEMPLATE = {
        "swagger": "2.0",
        "info": {
            "title": "DAIMON2 Assessment API",
            "description": "To be written",
            "contact": {
                "responsibleOrganization": "Clausthal",
                "responsibleDeveloper": "Awais Mushtaq",
                "email": "amus17@tu-clausthal.de",
                "url": "https://www.tu-clausthal.de/",
            },
            "termsOfService": "https://www.tu-clausthal.de/",
            "version": "0.0.1"
        },

        # comment for local environment
        # "host": "mdbdev.in.tu-clausthal.de",  # overrides localhost:500
        # "basePath": "/assessment-models",  # base bash for blueprint registration
        # "schemes": 'https',

        "operationId": "getmyData"
    }

    # MONGO_DB_USERNAME = "mdbadmin"
    # MONGO_DB_PASSWORD = "69bPc2%?LpP(g5"
    # MONGO_DB_AUTHSOURCE = "dss"


class ProductionConfig(Config):
    """
    This class will inherit any attributes from the parent Config class.
    Use this class to define production configuration atrributes, such
    as database usernames, passwords, server specific files & directories etc.
    """
    CACHE_API = 1
    STORAGE_DIRECTORY = os.path.join(app.root_path, "storage")


class DevelopmentConfig(Config):
    """
    This class will inherit any attributes from the parent Config class.
    Use this class to define development configuration atrributes, such
    as local database usernames, passwords, local specific files & directories etc.
    """
    DEBUG = True
    CACHE_API = 1
    STORAGE_DIRECTORY = os.path.join(app.root_path, "storage")
