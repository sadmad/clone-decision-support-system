

class Model:

    def __init__(self):
        self.set_name(None)
        self.set_trained_model(None)
        self.set_accuracy(None)

    def get_name(self):
        return self.__name
    def set_name(self, name):
        self.__name = name




    def get_trained_model(self):
        return self.__trained_model
    def set_trained_model(self, trained_model):
        self.__trained_model = trained_model





    def get_accuracy(self):
        return self.__accuracy

    def set_accuracy(self, accuracy):
        self.__accuracy = accuracy