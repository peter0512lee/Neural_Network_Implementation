import numpy as np

class NN():
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, activation):
        # input layer param
        self.W1 = np.random.randn(input_size, hidden_size_1) / np.sqrt(input_size)
        self.b1 = np.zeros([1, hidden_size_1])
        # hidden layer 1 param
        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) / np.sqrt(hidden_size_1)
        self.b2 = np.zeros([1, hidden_size_2])
        # hidden layer 2 param
        self.W3 = np.random.randn(hidden_size_2, output_size) / np.sqrt(hidden_size_2)
        self.b3 = np.zeros([1, output_size])

        self.activation = activation
        self.placeholder = {"x": None, "y": None}

    # Feed Placeholder
    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()

    # Forward Propagation
    def forward(self):
        n = self.placeholder["x"].shape[0]
        self.a1 = self.placeholder["x"].dot(self.W1) + np.ones((n, 1)).dot(self.b1)
        self.h1 = np.maximum(self.a1, 0)  # ReLU Activation
        self.a2 = self.h1.dot(self.W2) + np.ones((n, 1)).dot(self.b2)
        self.h2 = np.maximum(self.a2, 0)  # ReLU Activation
        self.a3 = self.h2.dot(self.W3) + np.ones((n, 1)).dot(self.b3)

        # Linear Activation
        if self.activation == "linear": 
            self.y = self.a3.copy()
        # Softmax Activation
        elif self.activation == "softmax":
            self.y_logit = np.exp(self.a3 - np.max(self.a3, 1, keepdims=True)) 
            self.y = self.y_logit / np.sum(self.y_logit, 1, keepdims=True)
        # Sigmoid Activation
        elif self.activation == "sigmoid":
            self.y = 1.0 / (1.0 + np.exp(-self.a3))

        return self.y

    # Backward Propagation
    def backward(self):
        n = self.placeholder["y"].shape[0]
        self.grad_a3 = (self.y - self.placeholder["y"]) / n
        self.grad_b3 = np.ones((n, 1)).T.dot(self.grad_a3)
        self.grad_W3 = self.h2.T.dot(self.grad_a3)

        self.grad_h2 = self.grad_a3.dot(self.W3.T) 
        self.grad_a2 = self.grad_h2 * np.maximum(self.a2, 0)  
        self.grad_b2 = np.ones((n, 1)).T.dot(self.grad_a2)
        self.grad_W2 = self.h1.T.dot(self.grad_a2)

        self.grad_h1 = self.grad_a2.dot(self.W2.T) 
        self.grad_a1 = self.grad_h1 * np.maximum(self.a1, 0)             
        self.grad_b1 = np.ones((n, 1)).T.dot(self.grad_a1)
        self.grad_W1 = self.placeholder["x"].T.dot(self.grad_a1)

    # Update Weights
    def update(self, learning_rate=1e-3):
        self.W1 = self.W1 - learning_rate * self.grad_W1
        self.b1 = self.b1 - learning_rate * self.grad_b1
        self.W2 = self.W2 - learning_rate * self.grad_W2
        self.b2 = self.b2 - learning_rate * self.grad_b2
        self.W3 = self.W3 - learning_rate * self.grad_W3
        self.b3 = self.b3 - learning_rate * self.grad_b3

    # Loss Functions
    def computeLoss(self):
        loss = 0.0
        # Mean Square Error
        if self.activation == "linear":
            loss = 0.5 * np.square(self.y - self.placeholder["y"]).mean()
        # Softmax Cross Entropy
        elif self.activation == "softmax":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6)
            loss = np.sum(loss, 1).mean()
        # Sigmoid Cross Entropy
        elif self.activation == "sigmoid":
            loss = -self.placeholder["y"] * np.log(self.y + 1e-6) - (-self.placeholder["y"]) * np.log(1-self.y + 1e-6)
            loss = np.mean(loss)
        return loss
