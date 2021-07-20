import numpy as np
import matplotlib.pyplot as plt
import MNISTtools       # MNISTtools.py
import NeuralNetwork    # NeuralNetwork.py


def OneHot(y):
    # --------------------------------
    # todo (Digit Label to One-Hot Key)
    y_one_hot = np.eye(10, dtype=np.float32)[y]
    return y_one_hot
    # --------------------------------


def Accuracy(y, y_):
    # --------------------------------
    # todo (Compute Accuracy)
    y_digit = np.argmax(y, 1)
    y_digit_ = np.argmax(y_, 1)
    temp = np.equal(y_digit, y_digit_).astype(np.float32)
    return np.sum(temp) / float(y_digit.shape[0])
    # --------------------------------


if __name__ == "__main__":
    # Dataset
    MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")

    # Show Data and Label
    print(x_train[0])
    print(y_train[0])
    # plt.imshow(x_train[0].reshape((28, 28)), cmap='gray')
    # plt.show()

    # --------------------------------
    # todo (Data Processing)
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    y_train = OneHot(y_train)
    y_test = OneHot(y_test)
    # --------------------------------

    # --------------------------------
    # todo (Create NN Model)
    nn = NeuralNetwork.NN(784, 256, 10, "softmax")
    # --------------------------------

    # Training the Model
    loss_rec = []
    batch_size = 64
    for i in range(101):
        # --------------------------------
        # todo (Sample Data Batch)
        batch_id = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_id]
        y_batch = y_train[batch_id]
        # --------------------------------

        # --------------------------------
        # todo (Forward & Backward & Update)
        nn.feed({"x": x_batch, "y": y_batch})
        nn.forward()
        nn.backward()
        nn.update(1e-2)
        # --------------------------------

        # --------------------------------
        # todo (Loss)
        loss = nn.computeLoss()
        loss_rec.append(loss)
        # --------------------------------

        # --------------------------------
        # todo (Evaluation)
        batch_id = np.random.choice(x_test.shape[0], batch_size)
        x_test_batch = x_test[batch_id]
        y_test_batch = y_test[batch_id]
        nn.feed({"x": x_test_batch})
        y_test_out = nn.forward()
        acc = Accuracy(y_test_out, y_test_batch)

        if i % 100 == 0:
            print("\r[Iteration {:5d}] Loss={:.4f} | Acc={:.3f}".format(i, loss, acc))
        # --------------------------------

    nn.feed({"x": x_test})
    y_prob = nn.forward()

    total_acc = Accuracy(y_prob, y_test)    # modify
    print("Total Accuracy:", total_acc)

    plt.title('Loss Curve')
    plt.plot(loss_rec)
    plt.show()
