from app.structure import dss


class ModelFactory:

    @staticmethod
    def get_model(model_type):

        if model_type == 1:
            return dss.NeuralNetwork()

        elif model_type == 2:
            return dss.RandomForest()

        elif model_type == 3:
            return dss.LinearRegressionM()

        elif model_type == 4:
            return dss.LogisticRegressionM()
        elif model_type == 5:
            return dss.DeepNeuralNetwork()

        elif model_type == 6:
            return dss.DecisionTreeRegressor()
