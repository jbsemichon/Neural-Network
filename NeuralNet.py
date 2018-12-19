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
        while epoch != epochs:
            random.shuffle(trainingdata)
            minibatches = [trainingdata[k:k+minibatch_size] for k in range(0,len(trainingdata),minibatch_size)]
            for minibatch in minibatches:
                for data in minibatch:
                    activations = [[int(data[0]),int(data[1])]]
                    target = self.set_target(data[2])
                    count = 0
                    szs = []
                    for layer in range(1,num_layers):
                        layer_z = []
                        zs = []
                        for neuron in range(size[layer]):
                            z = 0
                            xs = activations[count]
                            ws = self.weights[count]
                            for i in range(len(ws)):
                                z += xs[i]*ws[i]
                            layer_z.append(z)
                            z += self.biases[count]
                            zs.append(round(self.activation_function(z),4))
                        activations.append(zs)
                        szs.append(layer_z)
                        count += 1
                
        print("Epoch: %d" % (epoch))
        epoch += 1
    
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
            qcost.append(0.5* (cost ** 2))
            dcost.append(cost)
            count += 1
        total = 0
        for cost in qcost:
            total += cost
        return total,dcost

    def set_target(self,letter):
        if letter == "B":
            return  [0,1]
        else:
            return [1,0]

    def activation_function(self,z):
        #logistic function
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

epochs = 10
lrate = 2
trainingset_len = 100
testset_len = 20
minibatch_size = 10
trainingdata,test = extract_data(trainingset_len,testset_len)
size = [2,2,1]
num_layers = len(size)

network = Network(size,num_layers,epochs,lrate,trainingset_len,minibatch_size)
network.SGD(size,num_layers,trainingdata,minibatch_size,epochs)
