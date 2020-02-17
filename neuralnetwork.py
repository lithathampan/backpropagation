import numpy
def tanh(x):
    return (1.0 - numpy.exp(-2*x))/(1.0 + numpy.exp(-2*x))

def tanh_derivative(x):
    return (1 + tanh(x))*(1 - tanh(x))
def logistic(x):
    return 1.0/(1.0 + numpy.exp(-x))
def logistic_derivative(x):
    return (logistic(x)*(1-logistic(x)))
class NeuralNetwork:
    #########
    # parameters
    # ----------
    # self:      the class object itself
    # net_arch:  consists of a list of integers, indicating
    #            the number of neurons in each layer, i.e. the network architecture
    #########
    def __init__(self, net_arch):
        numpy.random.seed(0)
        
        # Initialized the weights, making sure we also 
        # initialize the weights for the biases that we will add later
        self.activity = tanh
        self.activity_derivative = tanh_derivative
        #self.activity = logistic
        #self.activity_derivative = logistic_derivative
        self.layers = len(net_arch)
        self.steps_per_epoch = 1
        self.arch = net_arch
        self.weights = []
        self.errors = []
        self.weighthistory = []
        self.w000 = []
        self.w001 = []
        self.w010 = []
        self.w011 = []
        self.w020 = []
        self.w021 = []
        self.w100 = []
        self.w110 = []
        self.w120 = []
        self.y01 = []
        self.y02 = []
        self.y11 = []
        self.y12 = []
        self.y2 = []
        self.iter = 0
        # Random initialization with range of weight values (-1,1)
        for layer in range(self.layers - 1):
            w = 2*numpy.random.rand(net_arch[layer] + 1, net_arch[layer+1]) - 1
            self.weights.append(w)
    
    def _forward_prop(self, x):
        y = x

        for i in range(len(self.weights)-1):
            activation = numpy.dot(y[i], self.weights[i])
            activity = self.activity(activation)

            # add the bias for the next layer
            activity = numpy.concatenate((numpy.ones(1), numpy.array(activity)))
            y.append(activity)

        # last layer
        activation = numpy.dot(y[-1], self.weights[-1])
        activity = self.activity(activation)
        y.append(activity)
        self.y01.append(y[0][1])
        self.y02.append(y[0][2])
        self.y11.append(y[1][1])
        self.y12.append(y[1][2])
        self.y2.append(y[2][0])
        return y
    
    def _back_prop(self, y, target, learning_rate):
        error = target - y[-1]
        self.errors.append(error)
        delta_vec = [error * self.activity_derivative(y[-1])]

        # we need to begin from the back, from the next to last layer
        for i in range(self.layers-2, 0, -1):
            error = delta_vec[-1].dot(self.weights[i][1:].T)
            error = error*self.activity_derivative(y[i][1:])
            delta_vec.append(error)

        # Now we need to set the values from back to front
        delta_vec.reverse()
        
        # Finally, we adjust the weights, using the backpropagation rules
        for i in range(len(self.weights)):
            layer = y[i].reshape(1, self.arch[i]+1)
            delta = delta_vec[i].reshape(1, self.arch[i+1])
            self.weights[i] += learning_rate*layer.T.dot(delta)
        #self.weighthistory.append({'w000':self.weights[0][0][0],'w001':self.weights[0][0][1],
        #'w010':self.weights[0][1][0],'w011':self.weights[0][1][1],
        #'w020':self.weights[0][2][0],'w021':self.weights[0][2][1],
        #'w100':self.weights[1][1][0],'w110':self.weights[1][1][0],'w120':self.weights[1][2][0]})
        self.w000.append(float(self.weights[0][0][0]))
        self.w001.append(float(self.weights[0][0][1]))
        self.w010.append(float(self.weights[0][1][0]))
        self.w011.append(float(self.weights[0][1][1]))
        self.w020.append(float(self.weights[0][2][0]))
        self.w021.append(float(self.weights[0][2][1]))
        self.w100.append(float(self.weights[1][0][0]))
        self.w110.append(float(self.weights[1][1][0]))
        self.w120.append(float(self.weights[1][2][0]))
    #########
    # parameters
    # ----------
    # self:    the class object itself
    # data:    the set of all possible pairs of booleans True or False indicated by the integers 1 or 0
    # labels:  the result of the logical operation 'xor' on each of those input pairs
    #########
    def fit(self, data, labels, learning_rate=0.1, epochs=100):
        
        # Add bias units to the input layer - 
        # add a "1" to the input data (the always-on bias neuron)
        ones = numpy.ones((1, data.shape[0]))
        Z = numpy.concatenate((ones.T, data), axis=1)
        
        for k in range(epochs):
            if (k+1) % 10000 == 0:
                print('epochs: {}'.format(k+1))
        
            sample = numpy.random.randint(data.shape[0])

            # We will now go ahead and set up our feed-forward propagation:
            x = [Z[sample]]
            y = self._forward_prop(x)
            # Now we do our back-propagation of the error to adjust the weights:
            target = labels[sample]
            self._back_prop(y, target, learning_rate)
            
    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    # 
    # parameters
    # ----------
    # self:   the class object itself
    # x:      single input data
    #########
    def predict_single_data(self, x):
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        for i in range(0, len(self.weights)):
            val = self.activity(numpy.dot(val, self.weights[i]))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return val[1]
    def predict_single_data_descriptive(self, x):
        print("Input:%s"%(x))
        val = numpy.concatenate((numpy.ones(1).T, numpy.array(x)))
        print("Concatenated:%s"%(val))
        for i in range(0, len(self.weights)):
            print("Left side:%s"%(val))
            print("Right side:%s"%(self.weights[i]))        
            print("Dotproduct:%s"%(numpy.dot(val, self.weights[i])))
            val = self.activity(numpy.dot(val, self.weights[i]))    
            print("Dotproduct_activity function:%s"%(val))
            val = numpy.concatenate((numpy.ones(1).T, numpy.array(val)))
        return val[1]
    #########
    # the predict function is used to check the prediction result of
    # this neural network.
    # 
    # parameters
    # ----------
    # self:   the class object itself
    # X:      the input data array
    #########
    def predict(self, X):
        Y = numpy.array([]).reshape(0, self.arch[-1])
        for x in X:
            y = numpy.array([[self.predict_single_data(x)]])
            Y = numpy.vstack((Y,y))
        return Y