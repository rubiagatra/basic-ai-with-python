import numpy as np
from lr_utils import load_dataset

class LogisticRegression:
    def __init__(self):
        self.params = None
        self.costs = None
        self.prediction = None
        
    def _initialize_weight(self, X):
        W = np.random.randn(X.shape[0], 1) * 0.01
        b = 0

        return {'W': W,
                'b': b}

    def _forward_propagation(self, X, params):
        W = params['W']
        b = params['b']

        Z = np.matmul(W.T, X) + b
        A = self._sigmoid(Z)

        return A

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _compute_cost(self, Y, A):
        m = Y.shape[1]
        cost = - (1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)))
        return cost

    def _backward_propagation(self, A, X, Y):
        m = X.shape[1]

        dW = 1/m * np.matmul(X, (A-Y).T)
        db = 1/m * np.sum(A-Y, keepdims=True, axis=1)

        return {'dW': dW,
                'db': db}


    def _update_parameters(self, grads, params, learning_rate):
        W = params['W']
        b = params['b']

        dW = grads['dW']
        db = grads['db']

        W = W - learning_rate * dW
        b = b - learning_rate * db

        return {'W':W,
                'b':b}




    def fit(self,X_train, y_train, num_iterations=2000, learning_rate=0.005, print_cost=False):
        params = self._initialize_weight(X_train)
        costs = []
        for i in range(num_iterations):
            A = self._forward_propagation(X_train, params)
            cost = self._compute_cost(y_train, A)
            grads = self._backward_propagation(A, X_train, y_train)
            params = self._update_parameters(grads, params, learning_rate)

            if print_cost and i % 100 == 0:
                print( "Iteration %i with cost: %f " % (i, cost))
                costs.append(cost)

        self.params = params
        self.costs = costs
        return params

    def predict(self, X_test):
        self.x_test = X_test
    
        m = X_test.shape[1]
        Y_prediction = np.zeros((1,m))
        W = self.params['W']
        b = self.params['b']

        A = self._sigmoid(np.dot(W.T, X_test) + b)

        for i in range(A.shape[1]):

            if A[0,i] > 0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0

        assert(Y_prediction.shape == (1, m))
    
        self.prediction = Y_prediction
        return self.prediction

    def accuracy(self, y_test):
        y_prediction_test = self.prediction
    
        return "Accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100)
    
    

def prepare_dataset():
    """
    Prepare datasets
    return --> X_train, y_train, X_test, y_test
    """
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T / 255
    return train_x_flatten, train_set_y, test_x_flatten, test_set_y

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = prepare_dataset()
    print("Dataset Example Shape")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print("Logistics Regression Example")
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train, print_cost=True)
    print(logreg.predict(X_test))
    print(logreg.accuracy(y_test))

