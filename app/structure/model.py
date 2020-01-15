

class ML_Model:

    def __init__(self):
        self.set_name(None)
        self.set_model(None)
        self.set_accuracy(None)

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

