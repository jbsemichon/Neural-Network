###Neural Network with 2 input nerons 2 hidden neurons and 2 output neurons

import random

##Global Constants

epochs = 2
e = 2.718
net_struc = [2,3,2,1] ##Make sure that you also change the number of layers otherwise you get index out of range errors!!!!!!
num_layers = len(net_struc)-1      ###Number of layers - 1
lr = 0.4

##Functions

def sigmoid(a):
    s = 1/(1 + (e**-a))
    return s

def derivative_of_sigmoid(value):
    return value - (1-value)

def generateweights(net_struc,choice):
    w = []
    for layer in range(num_layers):
        layer_w = []
        for neuroni in range(net_struc[layer]):
            neuroni_w = []
            for neuronj in range(net_struc[layer+1]):
                if choice == 0:
                    neuroni_w.append(round(random.uniform(-3,3),2))
                else:
                    neuroni_w.append(0)
            layer_w.append(neuroni_w)
        w.append(layer_w)
    return w        ##Returns an array with all of the weights in the form [[[w1,w2],[w3,w4]],[[w5,w6],[w7,w8]]] if choice == 0 otherwise all values are 0 based of a net_struc of [2,2,2]

def generatearray(net_struc):
    array = []
    for layer in range(num_layers):
        layer_array = []
        for layeri in range(net_struc[layer+1]):
##            layer_array.append(random.randint(2,98))
            layer_array.append(0)
        array.append(layer_array)
    return array        ##Returns an array with all values of 0 in the form [[[0,0],[0,0]],[[0,0],[0,0]]] based of a net_struc of [2,2,2]
        
def calc_error():
    error_total = 0
    for neuron in range(net_struc[num_layers]):
        activations[num_layers][neuron] = predicted_value
        expected[neuron] = expected_value
        error_total += (predicted_value - expected_value)**2  ##If you get an "Index out of range error" here make sure that there are enough expected
    print("Error:",error_total * 0.5)                                           ##results compared to the number of neurons in the output layer.
    return error_total * 0.5

def derivatives(layer,a_changes):     ##Layer == num_layer and goes backwards from num_layers to 0
    if layer == num_layers:
        for neuron in range(net_struc[layer]):
            print("activations[layer][neuron]",activations[layer][neuron])
            a_changes[layer][neuron] = round((activations[layer][neuron] - expected[neuron]) * round(derivative_of_sigmoid(activations[layer][neuron]),4),4) * activations[layer][neuron] ##If expected is 0 a_changes will be the same as activations
    elif layer != num_layers:
    ##Derivatives of previous activations
        for neuron in range(net_struc[layer]):
            a_changes[layer][neuron] = round(round(derivative_of_sigmoid(activations[layer][neuron]),4)*activations[layer][neuron],4)
            
    ##Derivatives of weights
    for neuron in range(net_struc[layer-1]):
        for node in range(net_struc[layer]):        ##w_changes[layer-1] because their are no weigths for the first layer therefore w_chagnes is smaller than num_layers by 1
            w_changes[layer-1][neuron][node] += round(a_changes[layer][node] * activations[layer-1][neuron],4)

    
    return 0

def change_weights(w_changes,layer,w):
    number1 = 0
    for strata in w_changes[layer-1]:
        number2 = 0
        for node in strata:
            w[layer-1][number1][number2] -= node * lr
            number2 += 1
        number1 += 1
    return w
    
##Other Variables
    
w = generateweights(net_struc,0)

##Main Body

for i in range(epochs):
    print("Epochs:",i+1)
    activations = generatearray(net_struc)
    w_changes = generateweights(net_struc,1)
    a_changes = generatearray(net_struc)
    
##    print("Weights:",w)
    
    #training_number = 0
    training_number = random.randint(0,3)
    if training_number == 0:
        activations.insert(0,[0,0])
        print("Input : [0,0]")
        expected = [0]
    elif training_number == 1:
        activations.insert(0,[1,0])
        print("Input : [1,0]")
        expected = [0]
    elif training_number == 2:
        activations.insert(0,[0,1])
        print("Input : [0,1]")
        expected = [0]
    elif training_number == 3:
        activations.insert(0,[1,1])
        print("Input : [1,1]")
        expected = [1]
    a_changes.insert(0,[0,0])
    
    
###Forward Pass
    for layer in range(num_layers):
        for neuroni in range(net_struc[layer+1]): ###Neuron in next layer
            total = 0
            for neuronj in range(net_struc[layer]): ###Neuron in layer
                total += w[layer][neuronj][neuroni] * activations[layer][neuronj]
            activations[layer+1][neuroni] = round(sigmoid(total),3)
    print("Output: ",activations[num_layers])
    print("Expected: ",expected)
    print("Activations",activations)
##Calculate Error
    error = round(calc_error(),3)
    
##Backwards Pass
    for layer in range(num_layers,0,-1):    ##Goes from num_layers to 0 (not inclusive)
        derivatives(layer,a_changes)
    print("w_changes:",w_changes)
    print("a_changes",a_changes)

    for layer in range(num_layers,0,-1): 
        change_weights(w_changes,layer,w)
    print("*******************************************************")


    















