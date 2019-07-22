###Neural Network with 2 input nerons 2 hidden neurons and 2 output neurons

import random

##Global Constants

epochs = 1000
e = 2.718
net_struc = [2,4,2]   ##Structure of network [input neurons,hiddenlayer1,hlayer2,...,output neurons]
num_layers = len(net_struc)-1      ###Number of layers - 1
lr = 0.5

##Functions

def sigmoid(a):
    s = 1/(1 + (e**-a))
    return s

def derivative_of_sigmoid(value):
    return value * (1-value)

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

def generatebiases(net_struc,choice):
    b = []
    for i in net_struc[1:]:
        temp_b = []
        for j in range(i):
            if choice == 0:
                temp_b.append(round(random.uniform(-3,3),2))
            elif choice == 1:
                temp_b.append(0)
        b.append(temp_b)
    return b

def forward_pass(num_layers,w,activations):
    for layer in range(num_layers):
        for neuroni in range(net_struc[layer+1]): ###Neuron in next layer
            total = 0
            for neuronj in range(net_struc[layer]): ###Neuron in layer
                total += w[layer][neuronj][neuroni] * activations[layer][neuronj]
            activations[layer+1][neuroni] = sigmoid(total+b[layer][neuroni])
    return activations

def calc_error(activations):
    error_total = 0
    for neuron in range(net_struc[num_layers]):
        predicted_value = activations[num_layers][neuron]
        expected_value = expected[neuron]
        error_total += ((expected_value - predicted_value)**2) * 0.5
        ##If you get an "Index out of range error" here make sure that there are enough expected
        ##results compared to the number of neurons in the output layer.
    return error_total

def derivatives(layer,a_changes,w_changes,b_changes):     ##Layer == num_layer and goes backwards from num_layers to 0
    if layer == num_layers:
        a_changes = derivative_of_first_layer(layer,a_changes)
    elif layer != num_layers:
        a_changes = derivative_of_other_layers(layer,a_changes)
    w_changes = derivatives_of_weights(layer,a_changes,w_changes)
    b_changes = derivatives_of_biases(layer,a_changes,b_changes)
    return 0

def derivative_of_first_layer(layer,a_changes):
    for neuron in range(net_struc[layer]):
        a_changes[layer-1][neuron] = -1*(expected[neuron]-activations[layer][neuron]) * derivative_of_sigmoid(activations[layer][neuron])
    return a_changes
    
def derivative_of_other_layers(layer,a_changes):
    for neuron in range(net_struc[layer]):
        for node in range(net_struc[layer+1]):
            a_changes[layer-1][neuron] += a_changes[layer][node] * w[layer][neuron][node]
    for neuron in range(net_struc[layer]):
        a_changes[layer-1][neuron] = derivative_of_sigmoid(a_changes[layer-1][neuron])
    return a_changes

def derivatives_of_weights(layer,a_changes,w_changes):
    for neuron in range(net_struc[layer-1]):
            for node in range(net_struc[layer]):        ##w_changes[layer-1] because their are no weigths for the first layer therefore w_chagnes is smaller than num_layers by 1
                w_changes[layer-1][neuron][node] += a_changes[layer-1][node] * activations[layer-1][neuron]
    return w_changes

def derivatives_of_biases(layer,a_changes,b_changes):
    for neuron in range(net_struc[layer]):
        b_changes[layer-1][neuron] += a_changes[layer-1][neuron] ## * 1
    return b_changes

def change_weights(w_changes,layer,w):
    number1 = 0
    for strata in w_changes[layer-1]:
        number2 = 0
        for node in strata:
            w[layer-1][number1][number2] -= node * lr
            number2 += 1
        number1 += 1
    return w

def change_biases(b_changes,layer,b,num_layers):
    for strata in range(net_struc[layer]):
        b_changes[layer-1][strata] = a_changes[layer-1][strata] * lr
    return b

##Other Variables

##w = [[[0.15,0.2,0.3,0.4,0.25],[0.05,0.35,0.30,0.2,0.30]],[[0.40,0.50],[0.40,0.50],[0.40,0.50],[0.40,0.50],[0.45,0.55]],[[0.3],[0.5]]]
##b = [[0.35,0.35,0.35,0.35,0.35],[0.60,0.60],[0.5]]

##w = [[[0.15,0.25],[0.20,0.30]],[[0.40,0.50],[0.45,0.55]]]
##b = [[0.35,0.35],[0.60,0.60]]

w = generateweights(net_struc,0)
b = generatebiases(net_struc,0)

##Main Body

for i in range(epochs):
    print("Epochs:",i+1)
    activations = generatearray(net_struc)
    w_changes = generateweights(net_struc,1)
    a_changes = generatearray(net_struc)
    b_changes = generatebiases(net_struc,1)

##    print("W",w)
##    print("B",b)

    activations.insert(0,[0.05,0.1])
    expected = [0.01,0.99]
    
    #training_number = 0
##    training_number = random.randint(0,3)
##    if training_number == 0:
##        activations.insert(0,[0.01,0.01])
##        expected = [0.01]
##    elif training_number == 1:
##        activations.insert(0,[.99,0.01])
##        expected = [0.01]
##    elif training_number == 2:
##        activations.insert(0,[0.01,0.99])
##        expected = [0.01]
##    elif training_number == 3:
##        activations.insert(0,[0.99,0.99])
##        expected = [0.99]
    
    
###Forward Pass
    
    activations = forward_pass(num_layers,w,activations)
##    print("Output: ",activations[num_layers])
##    print("Expected: ",expected)
##    print("Activations",activations[num_layers])
    
##Calculate Error
    
    error = calc_error(activations)
    print("Error:",error)
    
##Backwards Pass
    
    for layer in range(num_layers,0,-1):    ##Goes from num_layers to 0 (not inclusive)
        derivatives(layer,a_changes,w_changes,b_changes)
    for layer in range(num_layers,0,-1):
        change_weights(w_changes,layer,w)
        change_biases(b_changes,layer,b,num_layers)
##    print("w_changes",w_changes)
##    print("b_changes",b_changes)
##    print("a_changes",a_changes)
##    print("w",w)
##    print("*******************************************************")



##Test Neural Network
while True:
    choice = input("Do you want to test?")
    if choice == "y":
        test = []
        for i in range(net_struc[0]):
            test.append(float(input("Enter test data\n")))
        activations = generatearray(net_struc)
        activations.insert(0,test)
        activations = forward_pass(num_layers,w,activations)
        error = calc_error(activations)
        print("Error of test data:",error)
        print("Activations of neurons for test data:",activations[num_layers])
    else:
        break
        

    

    










