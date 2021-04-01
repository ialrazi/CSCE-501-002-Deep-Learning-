import numpy as np
import itertools
import torch.nn as nn
import torch


class Sigmoid:
    # Magic. Do not touch.
    def __init__(self):
        pass

    def forward(self, x):
        self.res = 1/(1+np.exp(-x))
        return self.res

    def backward(self):
        return self.res * (1-self.res)

    def __call__(self, x):
        return self.forward(x)


class Tanh:
    # Magic. Do not touch.
    def __init__(self):
        pass

    def forward(self, x):
        self.res = np.tanh(x)
        return self.res

    def backward(self):
        return 1 - (self.res**2)

    def __call__(self, x):
        return self.forward(x)


class GRU_Cell:
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
		
        self.Wzh = np.random.randn(h, h)
        self.Wrh = np.random.randn(h, h)
        self.Wh = np.random.randn(h, h)
        self.Wzx = np.random.randn(h, d)
        self.Wrx = np.random.randn(h, d)
        self.Wx = np.random.randn(h, d)
  
        self.r_act = Sigmoid()
        self.h_act = Tanh()
        self.z_act=Sigmoid()
    

        
        

    def forward(self, x_t, h_t):
	
        self.xt = x_t.reshape(-1,1) # reshape used for Matrix Transpose
        self.h_t = h_t.reshape(-1,1)
        self.zt= self.z_act(np.dot(self.Wzx, self.xt) + np.dot(self.Wzh, self.h_t)) #h_t=h_t-1
        self.rt= self.r_act(np.dot(self.Wrx, self.xt) + np.dot(self.Wrh, self.h_t)) #h_t=h_t-1
        self.z7=np.multiply(self.rt,self.h_t) # eqn (13)
        self.h_hat_t= self.h_act(np.dot(self.Wx, self.xt) + np.dot(self.Wh, np.multiply(self.rt, self.h_t)))
        self.h_t1= np.multiply(self.zt, self.h_t) + np.multiply((1 - self.zt), self.h_hat_t)
        return self.h_t1

    def backward(self, delta):
        
        
        dz_t = np.multiply(-delta,self.h_hat_t) #eqn (ii)
        dh_hat_t = np.multiply(delta,(1-self.zt)) #eqn (iii)
        dz_t += np.multiply(delta,self.h_t) #eqn (iv)
        dh_t = np.multiply(delta,self.zt )   #eqn (v)   
        dz10=np.multiply(dh_hat_t,self.h_act.backward()) #eqn (vi)
        dz8=dz9=dz10 # eqn (vii)
        dWx=np.dot(dz10,self.xt.T) #eqn(viii)
        dxt=np.dot(self.Wx.T,dz9) # eqn (ix)
        dWh=np.dot(dz8,self.z7.T) #eqn (x)
        dz7=np.dot(self.Wh.T,dz8) #eqn (xi)
        drt=np.multiply(dz7,self.h_t) #eqn (xii)
        dh_t+=np.multiply(dz7,self.rt) # eqn (xiii)
        dz6=np.multiply(drt,self.r_act.backward()) # eqn (xiv)
        dz4=dz5=dz6 # eqn (xv)
        dWrx=np.dot(dz5,self.xt.T) # eqn (xvi)
        dxt+=np.dot(self.Wrx.T,dz5) # eqn (xvii)
        dWrh=np.dot(dz4,self.h_t.T) # eqn (xviii)
        dh_t+=np.dot(self.Wrh.T,dz4) # eqn (xix)
        dz3=np.multiply(dz_t,self.z_act.backward()) # eqn (xx)
        dz1=dz2=dz3 # eqn(xxi)
        dWzx=np.dot(dz2,self.xt.T) # eqn (xxii)
        dxt+= np.dot(self.Wzx.T,dz2) # eqn (xxiii)
        dWzh=np.dot(dz1,self.h_t.T) # eqn (xxiv)
        dh_t+=np.dot(self.Wzh.T,dz1) # eqn (xxv)
        self.dWh=dWh
        self.dWx=dWx 
        self.dWrh=dWrh 
        self.dWrx=dWrx 
        self.dWzh=dWzh 
        self.dWzx=dWzx 
        self.dx = dxt 
        self.dh_t=dh_t 

        return self.dx,self.dh_t


# For your reference the test function is given below:		
def test():
    
    np.random.seed(11785)
    
    input_dim = 5
    
    hidden_dim = 2
    
    seq_len = 10
    
    data = np.random.randn(seq_len, input_dim)
    
    
    r1 = Tanh()(data)
    
    r2 = nn.Tanh()(torch.Tensor(data))
    
    g1 = GRU_Cell(input_dim, hidden_dim) # your
    
    g2 = nn.GRUCell(input_dim, hidden_dim, bias=False)
    
    hidden = np.random.randn(hidden_dim)
    
    
    
    g2.weight_ih = nn.Parameter(torch.cat([torch.Tensor(g1.Wrx), torch.Tensor(g1.Wzx), torch.Tensor(g1.Wx)]))
        
    g2.weight_hh = nn.Parameter(torch.cat([torch.Tensor(g1.Wrh), torch.Tensor(g1.Wzh), torch.Tensor(g1.Wh)]))
  
    o1=g1.forward(data[0],hidden).reshape(-1)
    
    print ("My_Forward",o1)
                                
    torch_data = torch.autograd.Variable(torch.Tensor(data[0]).unsqueeze(0), requires_grad=True)
                                
    torch_hidden = torch.autograd.Variable(torch.Tensor(hidden).unsqueeze(0), requires_grad=True)
                                
    o2 = g2.forward(torch_data, torch_hidden)
    print ("Torch_Forward",o2)
    
                                
    delta = np.random.randn(hidden_dim)
    
    o2.backward(torch.Tensor(delta).unsqueeze(0))
    
    delta = delta.reshape(-1, 1)
    
    dx,dh_t=g1.backward(delta) 
    dx_t = torch_data.grad
    dh=torch_hidden.grad
    
    g1_ih_grad = np.concatenate([g1.dWrx, g1.dWzx, g1.dWx], axis=0)
    
    g1_hh_grad = np.concatenate([g1.dWrh, g1.dWzh, g1.dWh], axis=0)
    
    g2_dWrx = g2.weight_ih.grad[0:hidden_dim, :]
    
    g2_dWzx = g2.weight_ih.grad[hidden_dim:2*hidden_dim, :]
    
    g2_dWx = g2.weight_ih.grad[2*hidden_dim:3*hidden_dim, :]
                                
    g2_dWrh = g2.weight_hh.grad[0:hidden_dim, :]
    
    g2_dWzh = g2.weight_hh.grad[hidden_dim:2*hidden_dim, :]
                                
    g2_dWh = g2.weight_hh.grad[2*hidden_dim:3*hidden_dim, :]
    
	print ("My_dx", g1.dx)
    print ("Torch_dx",dx_t)
    print ("My_dh", g1.dh_t)
    print ("Torch_dh",dh)     

	
test()



