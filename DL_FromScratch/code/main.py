import mnistLoader
from network import Sequential
import layers
import numpy as np
import doodler_forall as doodler


def run_mnist(images = 200, epochs=2, verbose=False):
    mnistImages, mnistLabels = mnistLoader.getFlattenedNormalizedMnist(
        count=images, visualize=True)    

    # Define the network architecture
    model = Sequential([
        layers.Dense(output_size=64, activation='relu', input_size=784),
        layers.Dense(output_size=10, activation='softmax', input_size=64)
    ])

    # Perform a forward pass through the network
    output = model.forward(mnistImages[0])

    print(f'Images shape:', mnistImages.shape)
    print(f'Labels len:', len(mnistLabels))
    print(f'First image shape:', mnistImages[0].shape)
    print(f'First label:', mnistLabels[0])
    print("Network output shape:", output.shape)

    print(model.fit(mnistImages, mnistLabels, epochs=epochs))

if __name__ == '__main__':
    print('Hello World!')

    # run_mnist(images=200)


    
    
    #doodler = Doodler()
    imageSet = doodler.gen_standard_cases(count=1000,rows=20,cols=20,
                                     wr=[0.2,0.5],hr=[0.2,0.4],
                                     noise=0, cent=False, show=False,
                                     flat=True,fc=(1,1),auto=False,
                                     mono=True,one_hots=True,multi=False)
    # THis is whats returned: (images, targets, labels, 2d-image-dimensions, flat)
    doodlerImages, doodlerTargets = imageSet[0], imageSet[1]
    print(f'Images.shape: {imageSet[0].shape}')
    print(f'Targets.shape: {imageSet[1].shape}')
    print(f'Target[0]: {imageSet[1][0]}')
    print(f'Labels: {len(imageSet[2])}')
    print(f'Labels[0]: {imageSet[2][0]}')
    print(f'2D Image Dimensions: {imageSet[3]}')
    print(f'Flat: {imageSet[4]}')

    # Define the network architecture
    model = Sequential([
        layers.Dense(output_size=64, activation='relu', input_size=20*20),
        layers.Dense(output_size=9, activation='softmax', input_size=64)],
        learning_rate=0.05,
        lossFunction='cross_entropy',
        regularization='L1',
        regLambda=0.01)
    print(f'Image1.shape: {imageSet[0][0].shape}')
    # Perform a forward pass through the network
    output = model.forward(imageSet[0][0])
    print("Network output:", output)

    print(f'Labels type: {type(imageSet[2])}')
    print("Network output type:", type(output))


    print(model.fit(doodlerImages, doodlerTargets, epochs=30, batchSize=100))



