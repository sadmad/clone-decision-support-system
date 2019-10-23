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
    NEURAL_NETWORK_MODEL = 'neural_network_finalized_model.sav'


class ProductionConfig(Config):
    """
    This class will inherit any attributes from the parent Config class.
    Use this class to define production configuration atrributes, such
    as database usernames, passwords, server specific files & directories etc.
    """
    NEURAL_NETWORK_MODEL = {
        'model':'neural_network_finalized_model.sav',
        'scaler':'nn_scaler.save'
    }

    RANDOM_FOREST_CLASSIFIER_MODEL = {
        'model':'random_forest_classifier_model.sav',
        'scaler':'rfc_scaler.save'
    }

    LINEAR_REGRESSION_MODEL = {
        'model':'linear_regression_model.sav',
        'scaler':'lr_scaler.save'
    }

    LOGISTIC_REGRESSION_MODEL = {
        'model':'logistic_model.sav',
        'scaler':'logr_scaler.save'
    }

class DevelopmentConfig(Config):
    """
    This class will inherit any attributes from the parent Config class.
    Use this class to define development configuration atrributes, such
    as local database usernames, passwords, local specific files & directories etc.
    """
    DEBUG = True