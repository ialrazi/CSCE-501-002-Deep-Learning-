import numpy as np
np.random.seed(1)

class Neural_Network(object):
    def __init__(self):        
        #Define Hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 2
        
        #Weights (parameters)
        self.init_weights()
        self.init_bias()
        
        
    def init_weights(self):
        
     
        self.Wh = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        self.Wo = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)
        
    def init_bias(self):
        self.Bh = np.full((1, self.hiddenLayerSize), np.random.rand())
        self.Bo = np.full((1, self.outputLayerSize), np.random.rand())
    
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    def sigmoid_prime(self,Z):
        return self.sigmoid(Z)*(1-self.sigmoid(Z))
    def ReLu(self,Z):
        return Z*(Z>=0)
    def ReLu_prime(self,Z):
        return 1*(Z>=0)
        
    def Tanh(self,Z):
        return np.tanh(Z)
    def Tanh_prime(self,Z):
        return 1.0 - np.tanh(Z)**2
        
    def activation_function(self,opt,Z):
        if opt==1:
            return self.sigmoid(Z)
        elif opt==2:
            return self.ReLu(Z)
        elif opt==3:
            return self.Tanh(Z)
    def activation_function_prime(self,opt,Z):
        if opt==1:
            return self.sigmoid_prime(Z)
        elif opt==2:
            return self.ReLu_prime(Z)
        elif opt==3:
            return self.Tanh_prime(Z)
    
    def feed_forward(self,X,opt):
        '''
        X    - input matrix
        opt  - option for activation function (1:sigmoid, 2:ReLu, 3:Tanh)
        Zh   - hidden layer weighted input
        Zo   - output layer weighted input
        H    - hidden layer activation
        y    - output layer
        yHat - output layer predictions
        '''

        # Hidden layer
        self.Zh = np.dot(X, self.Wh) + self.Bh
        self.H = self.activation_function(opt,self.Zh)
        
        # Output layer
        self.Zo = np.dot(self.H, self.Wo) + self.Bo
        
        self.yHat = self.activation_function(opt,self.Zo)
        
    def costFunctionPrime(self, X, y,opt,choice):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.feed_forward(X,opt)
        
        if choice==1:
            div_loss=(y - self.yHat)
        elif choice==2:
            div_loss=np.zeros(self.yHat.shape)
            div_loss[y==1]=-1/self.yHat[y==1]
            div_loss[y==0]=1/(1-self.yHat[y==0])
            
        
        dJdWo = np.dot(self.H.T,(div_loss* self.activation_function_prime(opt,self.Zo)))
        dJdWh = np.dot(X.T,  (np.dot(div_loss * self.activation_function_prime(opt,self.Zo), self.Wo.T) * self.activation_function_prime(opt,self.Zh)))
        dJdbh = np.sum((np.dot((div_loss*self.activation_function_prime(opt,self.Zo)),self.Wo.T))* self.activation_function_prime(opt,self.Zh))
        dJdbo = np.sum(div_loss*self.activation_function_prime(opt,self.Zo)) 
        
        return dJdWh, dJdWo,dJdbh,dJdbo
                             
    def backprop(self,X, y,lr,choice,opt):
        dWh, dWo, dBh, dBo = self.costFunctionPrime(X, y,opt,choice)
        self.Wh = self.Wh + dWh * lr
        self.Wo = self.Wo + dWo * lr
        self.Bh = self.Bh + dBh * lr
        self.Bo = self.Bo + dBo * lr
        

#if __name__== "main":
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

lr=0.1
iters=2
opts=[1,2,3]
choices=[1,2]
for j in range(len(opts)):
    NN=Neural_Network()
    opt=opts[j]
    for k in range(len(choices)):
        choice=choices[k]
        print "OPT:",opt
        print "Choice:",choice
        
        for i in range(iters):
            NN.feed_forward(X,opt)
            NN.backprop(X,y,lr,choice,opt)
        
            print "After Iteration",i+1
            print "Hidden_Layer_Weights",NN.Wh
            print "Hidden_Layer_bias",NN.Bh
            print "Output_Layer_Weights",NN.Wo
            print "Output_Layer_bias",NN.Bo
            print "Output", NN.yHat

        NN.feed_forward(X,opt)
        print "Hidden_Layer_Weights",NN.Wh
        print "Hidden_Layer_bias",NN.Bh
        print "Output_Layer_Weights",NN.Wo
        print "Output_Layer_bias",NN.Bo
        print "Output", NN.yHat
