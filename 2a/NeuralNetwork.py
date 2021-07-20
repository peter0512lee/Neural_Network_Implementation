import numpy as np

class NN():
    def __init__(self, input_size, hidden_size, output_size, activation):
        # init encoder weights
        self.e_W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.e_b1 = np.zeros([1, hidden_size])

        # init decoder weights
        self.d_W1 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.d_b1 = np.zeros([1, output_size])

        self.activation = activation
        self.placeholder = {"x": None, "y": None}

    # Feed Placeholder
    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()

    # # Encoder
    # def encoder(self):


    # # Decoder
    # def decoder(self):


    # Forward Propagation
    def forward(self):
        n = self.placeholder["x"].shape[0]
        self.e_a1 = self.placeholder["x"].dot(self.e_W1) + np.ones((n, 1)).dot(self.e_b1)
        self.e_h1 = np.maximum(self.e_a1, 0)  # ReLU Activation

        self.d_a1 = self.e_h1.dot(self.d_W1) + np.ones((n, 1)).dot(self.d_b1)

        # Linear Activation
        if self.activation == "linear": 
            self.y = self.d_a1.copy()
        # Softmax Activation
        elif self.activation == "softmax":
            self.y_logit = np.exp(self.d_a1 - np.max(self.d_a1, 1, keepdims=True)) 
            self.y = self.y_logit / np.sum(self.y_logit, 1, keepdims=True)
        # Sigmoid Activation
        elif self.activation == "sigmoid":
            self.y = 1.0 / (1.0 + np.exp(-self.d_a1))

        return self.y

    # Backward Propagation
    def backward(self):
        n = self.placeholder["y"].shape[0]
        self.grad_d_a1 = (self.y - self.placeholder["y"]) / n
        self.grad_d_b1 = np.ones((n, 1)).T.dot(self.grad_d_a1)
        self.grad_d_W1 = self.e_h1.T.dot(self.grad_d_a1)
        self.grad_e_h1 = self.grad_d_a1.dot(self.d_W1.T) 
        self.grad_e_a1 = self.grad_e_h1 * np.maximum(self.e_a1, 0)             
        self.grad_e_b1 = np.ones((n, 1)).T.dot(self.grad_e_a1)
        self.grad_e_W1 = self.placeholder["x"].T.dot(self.grad_e_a1)

    # Update Weights
    def update(self, learning_rate=1e-3):
        self.e_W1 = self.e_W1 - learning_rate * self.grad_e_W1
        self.e_b1 = self.e_b1 - learning_rate * self.grad_e_b1
        self.d_W2 = self.d_W1- learning_rate * self.grad_d_W1
        self.d_b2 = self.d_b1 - learning_rate * self.grad_d_b1

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
