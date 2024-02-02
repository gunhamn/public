from mnist import MNIST
import numpy as np



def getFlattenedNormalizedMnist(count=60000, visualize=False):
    mndata = MNIST('C:\\Users\\maxgu\\Projects\\public\\DL_FromScratch\\mnist')
    images, labels = mndata.load_training()  # or mndata.load_testing()

    # Slice the images and labels to return only the specified count
    images = images[:count]
    labels = labels[:count]

    # Normalize the images
    images = np.array(images) / 255.0

    # Convert labels to one-hot encoding
    num_classes = 10  # MNIST has 10 classes, for digits 0 through 9
    labels_onehot = np.eye(num_classes)[labels]

    if visualize:
        print('Loaded MNIST data')
        print('Number of images:', len(images))
        print('Number of labels:', len(labels))
        print('Image size:', len(images[0]))
        #print(f'First image:', images[0])
    
    return images, labels_onehot
    