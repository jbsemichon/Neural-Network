###Neural Network with 2 input nerons 2 hidden neurons and 2 output neurons

import random


x = [1,0]
expected = [0,0]
##x = [0,1]
##expected = 0
##
##x = [1,1]
##expected = 1

epochs = 10
e = 2.718
def sigmoid(a):
    s = 1/(1 + (e**-a))
    return s

def derivative_of_sigmoid(value):
    return value - (1-value)

def generateweights(net_struc):
    w = []
    for layer in net_struc:
        layer_w = []
        for neuron in range(layer):
            layer_w.append(round(random.uniform(0,3),2))
        w.append(layer_w)
    return w

def generatearray(net_struc):
    array = []
    for layer in net_struc:
        layer_array = []
        for neuron in range(layer):
            layer_array.append(0)
        array.append(layer_array)
    return array

def calc_error():
    error_total = 0
    for neuron in range(net_struc[num_layers]):
        error_total += (activations[num_layers][neuron] - expected[neuron])**2
    return error_total * 0.5

    
net_struc = [2,2,2]
num_layers = 2      ###Number of layers - 1
w = generateweights(net_struc)
totals = generatearray(net_struc)
activations = generatearray(net_struc)
w_grad_func = generatearray(net_struc)




for i in range(epochs):
    dcosts = []
###Forward Pass

    for layer in range(num_layers + 1):
        total = 0
        for node in range(net_struc[layer]):
            for neuron in range(net_struc[num_layers]): 
                total += x[neuron] * w[layer][neuron]
            totals[layer][neuron] = total       ### = Because the array is created with all values set to 0
            activations[layer][neuron] = sigmoid(total)
            
###Calculate Error
            
    error = calc_error()
    
###Backwards Pass
    ###Derivative of all output neurons
    for neuron in range(net_struc[num_layers]):
        dcosts.append(activations[num_layers][neuron]- expected[neuron])
        
    ###First backwards pass

    ###Weight derivative
    print("W:",w)
    count = 0
    for neuron in range(net_struc[num_layers - 1]):
        w_grad_func[num_layers][count] = ### * derivative_of_sigmoid(activations[num_layers][count]) * w[num_layers][count]
        print("Derivative of sigmoid:",derivative_of_sigmoid(activations[num_layers][count]))
        print("Weight:",w[num_layers][count])
        print("Neuron:",neuron)
        count += 1
    print(w_grad_func)













    
            

            
        


            
