import mnistLoader
from network import Sequential
import layers
import numpy as np
import doodler_forall as doodler
from configparser import ConfigParser

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

def splitDataset(x, y, split= [0.7, 0.2, 0.1]):
    # Split the dataset into training, validation and test sets
    n = len(x)
    x_train = x[:int(n*split[0])]
    y_train = y[:int(n*split[0])]
    x_val = x[int(n*split[0]):int(n*(split[0]+split[1]))]
    y_val = y[int(n*split[0]):int(n*(split[0]+split[1]))]
    x_test = x[int(n*(split[0]+split[1])):]
    y_test = y[int(n*(split[0]+split[1])):]
    return x_train, y_train, x_val, y_val, x_test, y_test

def run_config(filePath):
    config = ConfigParser()
    config.read(filePath)
    parsed_config = {section: dict(config.items(section)) for section in config.sections()}
    
    print(parsed_config['DATASET']['split'])
    split = [float(x) for x in parsed_config['DATASET']['split'].split(',')]
    x_train, y_train, x_val, y_val, x_test, y_test = splitDataset(x, y, split= split)

    pass # This function is not in use

def generateDoodleCases(count=500, rows=50, cols=50, split=[0.7, 0.2, 0.1], print=False):
    if print:
        doodler.gen_standard_cases(count=5,rows=rows,cols=cols,
                                     types=["ball","triangle","box"],
                                     wr=[0.2,0.5],hr=[0.2,0.4],
                                     noise=0, cent=False, show=True,
                                     flat=True,fc=(1,1),auto=False,
                                     mono=True,one_hots=True,multi=False)
    
    trainCount = int(np.round(count*split[0]))
    valCount = int(np.round(count*split[1]))
    testCount = int(np.round(count*split[2]))

    trainImages = doodler.gen_standard_cases(count=trainCount,rows=rows,cols=cols,
                                     types=["ball","triangle","box"],
                                     wr=[0.2,0.5],hr=[0.2,0.4],
                                     noise=0, cent=False, show=False,
                                     flat=True,fc=(1,1),auto=False,
                                     mono=True,one_hots=True,multi=False)
    x_train, y_train = trainImages[0], trainImages[1]

    valImages = doodler.gen_standard_cases(count=valCount,rows=rows,cols=cols,
                                     types=["ball","triangle","box"],
                                     wr=[0.2,0.5],hr=[0.2,0.4],
                                     noise=0, cent=False, show=False,
                                     flat=True,fc=(1,1),auto=False,
                                     mono=True,one_hots=True,multi=False)
    x_val, y_val = valImages[0], valImages[1]

    testImages = doodler.gen_standard_cases(count=testCount,rows=rows,cols=cols,
                                     types=["ball","triangle","box"],
                                     wr=[0.2,0.5],hr=[0.2,0.4],
                                     noise=0, cent=False, show=False,
                                     flat=True,fc=(1,1),auto=False,
                                     mono=True,one_hots=True,multi=False)
    x_test, y_test = testImages[0], testImages[1]
    
    return x_train, y_train, x_val, y_val, x_test, y_test

def getConfigurationModels(rows, cols, outputSize):
    model1 = Sequential([
        layers.Dense(output_size=32, activation='relu', input_size=rows*cols),
        layers.Dense(output_size=3, activation='linear', input_size=32),
        layers.Dense(output_size=outputSize, activation='softmax', input_size=3)],
        learning_rate=0.1,
        lossFunction='cross_entropy',
        regularization='l2',
        regLambda=0.002)
    
    model2 = Sequential([
        layers.Dense(output_size=outputSize, activation='relu', input_size=rows*cols)],
        learning_rate=0.05,
        lossFunction='cross_entropy',
        regularization='l2',
        regLambda=0.02)
    
    model3 = Sequential([
        layers.Dense(output_size=64, activation='relu', input_size=rows*cols, w_range=[-0.1, 0.1], b_range=[-0.1, 0.1]),
        layers.Dense(output_size=32, activation='relu', input_size=64, w_range=[-0.1, 0.1], b_range=[-0.1, 0.1]),
        layers.Dense(output_size=16, activation='relu', input_size=32, w_range=[-0.1, 0.1], b_range=[-0.1, 0.1]),
        layers.Dense(output_size=8, activation='relu', input_size=16, w_range=[-0.1, 0.1], b_range=[-0.1, 0.1]),
        layers.Dense(output_size=outputSize, activation='softmax', input_size=8, w_range=[-0.1, 0.1], b_range=[-0.1, 0.1])],
        learning_rate=0.1,
        lossFunction='cross_entropy',
        regularization='l2',
        regLambda=0.0002)
    
    return model1, model2, model3


if __name__ == '__main__':
    rows,cols = 50, 50
    x_train, y_train, x_val, y_val, x_test, y_test = generateDoodleCases(
        count=500, rows=rows, cols=cols, split=[0.7, 0.2, 0.1], print=False)
    
    model1, model2, model3 = getConfigurationModels(rows, cols, len(y_train[0]))
    
    #model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batchSize=100)
    #print(f'Test set accuracy: {model1.accuracy(x_test, y_test)}')
    
    #model2.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=300, batchSize=100)
    #print(f'Test set accuracy: {model2.accuracy(x_test, y_test)}')
    
    model3.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=800, batchSize=100)
    print(f'Test set accuracy: {model3.accuracy(x_test, y_test)}')
    
