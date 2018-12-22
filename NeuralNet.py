import numpy
import random
import time

class Network(object):
    def __init__(self,size,num_layers,epochs,lrate,dataset_len,minibatch_size):
        self.size = size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lrate = lrate
        self.dataset_len = dataset_len
        self.minibatch_size = minibatch_size
        self.weights = self.set_weights(self.size,self.num_layers)
        self.biases  = self.set_biases(self.size,self.num_layers)
        print("Weights:",self.weights)
        print("Biases:",self.biases)
       
    def set_weights(self,size,num_layers):
        layer = 1
        weights = []
        for s in range(num_layers-1):
            for i in range(size[layer]):
                weights.append([round(random.uniform(0,0.5),2) for i in range(size[layer-1])])
            layer += 1
        return weights
    
    def set_biases(self,size,num_layers):
        layer = 1
        biases = []
        for i in range(num_layers-1):
            for i in range(size[layer]):
                biases.append(round(random.uniform(0,0.5),2))
            layer += 1
        return biases

    def SGD(self,size,num_layers,trainingdata,minibatch_size,epochs):
        epoch = 1
        while epoch != (epochs + 1):
            random.shuffle(trainingdata)
            minibatches = [trainingdata[k:k+minibatch_size] for k in range(0,len(trainingdata),minibatch_size)]
            for minibatch in minibatches:
                for data in minibatch:
                    activations = [[int(data[0]),int(data[1])]]
                    target = self.set_target(data[2])
                    count = 0
                    szs = []
                    for layer in range(1,num_layers):
                        array_of_weighted_sums = []
                        output = []
                        for neuron in range(size[layer]):
                            weighted_sum = 0
                            # Get input values
                            input_values = activations[count]
                            # Get weights
                            weight = self.weights[count]
                            for i in range(len(weight)):
                                # Calculate weightings
                                weighted_sum += input_values[i]*weight[i]
                            array_of_weighted_sums.append(weighted_sum)
                            # Add bias
                            weighted_sum += self.biases[count]
                            # Calculate activation function
                            output.append(round(self.activation_function(weighted_sum),4))
                        activations.append(output)
                        szs.append(array_of_weighted_sums)
                        count += 1
                
            print("Epoch: %d" % (epoch))
            epoch += 1
        print("Done!")
        print("Input values: "+str(activations[0][0])+", "+str(activations[0][1]))
        print("Output from activation function (Bias of "+ str(self.biases[0])+"): " + str(activations[1][0]) +" and " + str(activations[1][1]))
        print("Output from activation function (Bias of "+ str(self.biases[1])+"): " + str(activations[2][0]))
        print("Cost derivative: " + cost_derivative(activations, ))

    def sigmoidprime(self,z):
        return (self.sigmoid(z) - (1 - self.sigmoid(z)))
    
    def createlists(self,weight,bias):
        delta_w = []
        delta_b = []
        for i in weight:
            delta_w.append([])
        for i in bias:
            delta_b.append([])
        return delta_w, delta_b

    def cost_derivative(self,activation,y):
        qcost = []
        dcost = []
        count = 0
        for i in activation:
            cost = i-y[count]
            # Squared loss function
            qcost.append(0.5* (cost ** 2))
            dcost.append(cost)
            count += 1
        total = 0
        for cost in qcost:
            total += cost
        return total,dcost

    def set_target(self,letter):
        # Some weird conversion. Wouldn't this be more logically a bool???
        if letter == "B":
            return  [0,1]
        else:
            return [1,0]

    def activation_function(self,z):
        # Logistic function
        return 1/(1+numpy.exp(-z))



def extract_data(dataset_len,testset_len):
    file = open("data.txt","r")
    trainingdata = []
    for i in range(dataset_len):
       trainingdata.append(file.readline())
    testdata = []
    for i in range(testset_len):
        testdata.append(file.readline())
    return trainingdata,testdata

epochs = 1
lrate = 2
trainingset_len = 100
testset_len = 20
minibatch_size = 10
trainingdata,test = extract_data(trainingset_len,testset_len)
size = [2,2,1]
num_layers = len(size)

network = Network(size,num_layers,epochs,lrate,trainingset_len,minibatch_size)
network.SGD(size,num_layers,trainingdata,minibatch_size,epochs)
