import numpy as np

class LossFuncs():

    @staticmethod
    def categorical_cross_entropy(y_hat, y, epsilon=1e-12):

        y_hat_clipped = np.clip(y_hat, epsilon, 1. - epsilon)
        loss = -np.sum(y*np.log(y_hat_clipped), axis=0)

        return np.mean(loss) 

    