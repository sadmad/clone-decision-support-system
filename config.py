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
    MODELS = {

        'FDI_ASSESMENT': 'fdi_assesment_',  # FISH
        'CF_ASSESMENT': 'cf_assesment_',  # FISH
        'LHI_ASSESMENT': 'lhi_assesment_',  # FISH
        'MUSCEL_CWA_ASSESMENT': 'muscel_cwa_assessment_',  # FISH
        'LIVER_CWA_ASSESMENT': 'liver_cwa_assessment_',  # FISH
        'ERY_ASSESMENT': 'ery_assessment_',  # FISH
        'HB_ASSESMENT': 'hb_assessment_',  # FISH
        'GLU_ASSESMENT': 'glu_assessment_',  # FISH
        'HCT_ASSESMENT': 'hct_assessment_',  # FISH
        'GILL_CWA_ASSESMENT': 'gill_cwa_assessment_',  # FISH
    }

    MODELS_ID_MAPPING = {
        0: 'FDI_ASSESMENT',
        1: 'CF_ASSESMENT',
        2: 'LHI_ASSESMENT',
        3: 'MUSCEL_CWA_ASSESMENT',
        4: 'LIVER_CWA_ASSESMENT',
        5: 'ERY_ASSESMENT',
        6: 'HB_ASSESMENT',
        7: 'GLU_ASSESMENT',
        8: 'HCT_ASSESMENT',
        9: 'GILL_CWA_ASSESMENT',
    }

    EGEOS = {
        'base_url': 'https://www.amucad.org',
        'user_name': 'AMushtaq',
        'password': 'lK98hgr&h'
    }


class ProductionConfig(Config):
    """
    This class will inherit any attributes from the parent Config class.
    Use this class to define production configuration atrributes, such
    as database usernames, passwords, server specific files & directories etc.
    """
    CACHE_API = 1
    STORAGE_DIRECTORY = './storage/'


class DevelopmentConfig(Config):
    """
    This class will inherit any attributes from the parent Config class.
    Use this class to define development configuration atrributes, such
    as local database usernames, passwords, local specific files & directories etc.
    """
    DEBUG = True
    CACHE_API = 1
    STORAGE_DIRECTORY = './storage/'
