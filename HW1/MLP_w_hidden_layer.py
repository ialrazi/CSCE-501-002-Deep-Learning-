import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import torch.utils.data as data_utils
import torch
from torch import nn

def sample_points(n):
 # returns (X,Y), where X of shape (n,2) is the numpy array of points and Y is the array of classes
    radius = np.random.uniform(low=0,high=2,size=n).reshape(-1,1) # uniform radius between 0 and 2
    angle = np.random.uniform(low=0,high=2*np.pi,size=n).reshape(-1,1) # uniform angle
    x1 = radius*np.cos(angle)
    x2=radius*np.sin(angle)
    y = (radius<1).astype(int).reshape(-1)
    x = np.concatenate([x1,x2],axis=1)
    return x,y
def testing_routine(net,dataset):
  # Now for the validation set
    test_data,test_labels=dataset
    test_output = net(test_data)
    # compute the accuracy of the prediction
    test_prediction = test_output.cpu().detach().argmax(dim=1)
    test_accuracy = (test_prediction.numpy()==test_labels.numpy()).mean()
    print("Testing accuracy :",test_accuracy) 
    plot_points([test_data,test_prediction])

# Define training process
def training_routine(net,dataset,n_iters,gpu):
    # organize the data
    train_data,train_labels,val_data,val_labels = dataset
    #train,valiadation=dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    # use the flag
    if gpu:
        train_data,train_labels = train_data.cuda(),train_labels.cuda()
        val_data,val_labels = val_data.cuda(),val_labels.cuda()
        net = net.cuda() # the network parameters also need to be on the gpu !
        print("Using GPU")
    else:
        print("Using CPU")

    for i in range(n_iters):
        # forward pass

        train_output = net(train_data)
        train_loss = criterion(train_output,train_labels)
        # backward pass and optimization
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Once every 100 iterations, print statistics
        if i%100==0:
            print("At iteration",i)
            # compute the accuracy of the prediction
            train_prediction = train_output.cpu().detach().argmax(dim=1)
            train_accuracy = (train_prediction.numpy()==train_labels.numpy()).mean()
            # Now for the validation set
            val_output = net(val_data)
            val_loss = criterion(val_output,val_labels)
            # compute the accuracy of the prediction
            val_prediction = val_output.cpu().detach().argmax(dim=1)
            val_accuracy = (val_prediction.numpy()==val_labels.numpy()).mean()
            print("Training loss :",train_loss.cpu().detach().numpy())
            print("Training accuracy :",train_accuracy)
            print("Validation loss :",val_loss.cpu().detach().numpy())
            print("Validation accuracy :",val_accuracy)
    
def generate_dataset(training_points_num,validation_points_num,testing_points_num):
  
      # generating dataset
      train_data,train_labels= sample_points(training_points_num)
      train_data=torch.from_numpy(train_data).float()
      train_labels=torch.from_numpy(train_labels)


      val_data,val_labels= sample_points(validation_points_num)
      val_data=torch.from_numpy(val_data).float()
      val_labels=torch.from_numpy(val_labels)

      testing_data,testing_labels=sample_points(testing_points_num)
      testing_data=torch.from_numpy(testing_data).float()
      testing_labels=torch.from_numpy(testing_labels)

      #generating testing dataset
      dataset=[train_data,train_labels,val_data,val_labels]
      testing_dataset=[testing_data,testing_labels]

      return dataset,testing_dataset

class Net(torch.nn.Module):
      def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()
      
 
def plot_points(points):
      fig, ax = plt.subplots()
      ax.set_xlim((-2, 2))
      ax.set_ylim((-2, 2))
      colors=['red','green']
      X,Y=points[0],points[1]
      circle1 = plt.Circle((0, 0), 1, color='yellow',zorder=1)
      ax.add_artist(circle1)

      for i in range(len(X)):

        x,y=X[i]
        x=float(x)
        y=float(y)
        
        color=int(Y[i])
        
        circle=matplotlib.patches.Circle(xy=(x,y),radius=0.02,color=colors[color],zorder=2)
        ax.add_patch(circle)

      plt.show()

dataset,testing_dataset=generate_dataset(10000,2000,2000)
net=Net(2,128)
n_iters=1000
gpu=False
# call training routine
training_routine(net,dataset,n_iters,gpu)
testing_routine(net,testing_dataset)
