import os
import struct
import numpy as np
import matplotlib.pyplot as plt

mnist_path = "http://yann.lecun.com/exdb/mnist/"

zip_list = ["train-images-idx3-ubyte.gz", 
            "train-labels-idx1-ubyte.gz", 
            "t10k-images-idx3-ubyte.gz", 
            "t10k-labels-idx1-ubyte.gz" ]

ext_list = ["train-images.idx3-ubyte", 
            "train-labels.idx1-ubyte", 
            "t10k-images.idx3-ubyte", 
            "t10k-labels.idx1-ubyte" ]

def downloadMNIST(path = ".", unzip=True):
    import urllib.request
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(len(zip_list)):
        zip_name = zip_list[i]
        fname1 = os.path.join(path, zip_name)
        print("Download ", zip_name, "...")
        if not os.path.exists(fname1):
            urllib.request.urlretrieve( mnist_path + zip_name, fname1)
        else:
            print("pass")

        if unzip:
            import gzip
            import shutil
            ext_name = ext_list[i]
            fname2 = os.path.join(path, ext_name)
            print("Extract", fname1, "...")
            if not os.path.exists(fname2):
                with gzip.open(fname1, 'rb') as f_in:
                    with open(fname2, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                print("pass")
            

def loadMNIST(dataset = "training", path = "."):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl, dtype=np.int8)
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows*cols)
    fimg.close()

    return img, lbl

if __name__ == '__main__':
    downloadMNIST(path='MNIST_data', unzip=True)
    x_train, y_train = loadMNIST(dataset="training", path="MNIST_data")
    x_test, y_test = loadMNIST(dataset="testing", path="MNIST_data")

    print("[Visualize Training Data]")
    for i in range(3):
        print(y_train[i])
        plt.imshow(x_train[i].reshape([28,28]), cmap="gray")
        plt.show()

    print("[Visualize Testing Data]")
    for i in range(3):
        print(y_test[i])
        plt.imshow(x_test[i].reshape([28,28]), cmap="gray")
        plt.show()