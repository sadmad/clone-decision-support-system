
from app.structure import dss
from app import app
import os
import pandas as pd



class Finding:
    def __init__(self, model_type, model_config):
        print(' Findings Constructor')


class Fish(Finding):

    def __init__(self, model_type, model_config):

        print(' Fish Constructor')
        super().__init__( model_type, model_config )

        #Data Members
        self._features = [
            "Cultivator",
            "Alchol",
            "Malic_Acid",
            "Ash",
            "Alcalinity_of_Ash",
            "Magnesium",
            "Total_phenols",
            "Falvanoids",
            "Nonflavanoid_phenols",
            "Proanthocyanins",
            "Color_intensity",
            "Hue",
            "OD280",
            "Proline"
        ]






class FDI_ASSESMENT(Fish):
    
    def __init__(self, model_config):

        print(' FDI_ASSESMENT Constructor')
        super().__init__( model_type, app.config['MODELS']['FDI_ASSESMENT'] )


    def start(self):

        print(' FDI_ASSESMENT Start')
        



