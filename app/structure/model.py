

class ML_Model:

    def __init__(self, model_name):
        self.__name = None
        self.__model = model_name
        self.__accuracy = None
        self.saved_model_scaler = os.path.join(app_path, obj['scaler'])
        self.saved_model_path = os.path.join(app_path, obj['model'])

    def set_name(self, name):
        self.__name = name
    def set_model(self, model):
        self.__model = model
    def set_accuracy(self, accuracy):
        self.__accuracy = accuracy


    def get_model(self):
        return self.__model
    def get_name(self):
        return self.__name
    def get_accuracy(self):
        return self.__accuracy

class FDI_ASSESMENT(ML_Model):
    def data_intialization(self):
        print(' NeuralNetwork Return Model')
        return ''
    def main():
        return 'awais'
