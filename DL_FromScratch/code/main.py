import mnistLoader
from network import Sequential
import layers
import numpy as np


if __name__ == '__main__':
    print('Hello World!')

    mnistImages, mnistLabels = mnistLoader.getFlattenedNormalizedMnist(
        count=200, visualize=True)    

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

    print(f"Loss: {model.loss(np.array([4, 5]), np.array([5, 5]))}")

    print(model.fit(mnistImages, mnistLabels, epochs=1))


