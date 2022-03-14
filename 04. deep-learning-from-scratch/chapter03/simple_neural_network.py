import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        pass
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def identity_function(self, X):
        return X
    
    def init_network(self):
        network = {}
        network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2 x 3
        network['b1'] = np.array([[0.1, 0.2, 0.3]]) # 3 x 1
        network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 3 x 2
        network['b2'] = np.array([[0.1, 0.2]]) # 2 x 1
        network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2 x 2
        network['b3'] = np.array([[0.1, 0.2]]) # 2 x 1
        return network

    def forward(self, network, x):
        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']
        
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y  = self.identity_function(a3)
        return y

if __name__ == "__main__":
    NN =  SimpleNeuralNetwork()
    network = NN.init_network()
    x = np.array([1.0 ,0.5])
    y = NN.forward(network, x)
    print (y)
