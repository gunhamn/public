from mnist import MNIST

mndata = MNIST('path/to/mnist/data')
images, labels = mndata.load_training()
# or mndata.load_testing()
