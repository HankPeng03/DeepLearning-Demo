import numpy as np
from math  import exp
from numpy.random import randn

def activate(weights, inputs):
    activation = weights[-1] # the value of bias
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation

def transfer(activation):
    return sigmoid(activation)

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

def forward_propagate(networks, row):
    inputs = row

    for layer in networks[0]:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    new_inputs = []
    for neuron in networks[1]:
        activation = activate(neuron['weights'],inputs)
        neuron['output'] = transfer(activation)
        new_inputs.append(neuron['output'])
    inputs = new_inputs
    return inputs


def network(n_input, n_hidden, n_hidden_layer, n_output):

    networks = []
    networks.append([[{'weights':[randn() for i in range(n_input)]} for j in range(n_hidden)]
                     for k in range(n_hidden_layer)])
    networks.append([{'weights':[randn() for i in range(n_hidden)]} for j in range(n_output)])
    return networks


def backward_propagate_error(networks,expected):
    for i in reversed(range(len(networks))):
        layer = networks[i]
        errors = list()
        if i != len(networks)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in networks[i+1]: # the next layer
                    error += neuron['weights'][j]*neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j]*transfer(neuron['output'])

def update_weights(networks, row, l_rate):
    for i in range(len(networks)):
        inputs = row[:-1]
        if i!=0:
            inputs = [neuron['output'] for neuron in networks[i-1]]
        for neuron in networks[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate*neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
def train_network(networks, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            outputs = forward_propagate(networks,row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(networks,expected)
            update_weights(networks,row,l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))




if __name__ == '__main__':
    n_hidden = 3
    n_layer = 3
    n_input = 3

    np.random.seed(1)
    dataset = [[2.7810836,2.550537003,0],
        [1.465489372,2.362125076,0],
        [3.396561688,4.400293529,0],
        [1.38807019,1.850220317,0],
        [3.06407232,3.005305973,0],
        [7.627531214,2.759262235,1],
        [5.332441248,2.088626775,1],
        [6.922596716,1.77106367,1],
        [8.675418651,-0.242068655,1],
        [7.673756466,3.508563011,1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    networks = network(n_input=n_inputs, n_hidden=3, n_hidden_layer=3, n_output=n_outputs)
    train_network(network, dataset, 0.5, 20, n_outputs)
    for layer in network:
        print(layer)