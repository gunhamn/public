

def neuron(inputs, weights, bias, actFunction):
    total = sum(input * weight for input, weight in zip(inputs, weights)) + bias

    return actFunction(total)



