from app.structure import dss


class ModelFactory:

    def __init__(self):
        self.model = None

    def get_model(self, model_type):
        if model_type == 1:
            self.model = dss.NeuralNetwork()

        elif model_type == 2:
            self.model = dss.RandomForest()

        elif model_type == 3:
            self.model = dss.LinearRegressionM()

        elif model_type == 4:
            self.model = dss.LogisticRegression()
        elif model_type == 5:
            self.model = dss.DeepNeuralNetwork()

        elif model_type == 6:
            self.model = dss.DecisionTree()

        return self.model
