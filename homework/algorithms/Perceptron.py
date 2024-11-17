class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, train_x, train_y):
        # Initialize weights to zeros with the same length as the number of features
        self.weights = [0] * len(train_x[0])