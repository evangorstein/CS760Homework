from time import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor


#Pytorch and torchvision versions
print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

##Preliminaries
#Download the data
mnist_trainset = datasets.MNIST(root='..', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='..', train=False, download=True, transform=ToTensor())

#Data loader that we can use to read in the data a batch at a time
bs = 32
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=bs, shuffle=True)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=bs, shuffle=True)


##Learn about the data format
#Get a sample batch
images, labels = iter(trainloader).next()

#Print info about the batch
print(f"Shape of batch of images is {images.shape}")
print(f"And shape of corresponding labels is {labels.shape}")

#Show some of the images in the batch
figure = plt.figure()
num_of_images = 30
for index in range(1, 1+num_of_images):
    plt.subplot(3, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

plt.show()



##Build model myself
input_size = 784
hidden_sizes = [300, 200]
output_size = 10

class my_model():
    
    def __init__(self):
        
        #converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.W1 = 2 * torch.rand([hidden_sizes[0], input_size]) - 1
        self.b1 = 2 * torch.rand(hidden_sizes[0]) - 1

        self.W2 = 2 * torch.rand([hidden_sizes[1], hidden_sizes[0]]) - 1
        self.b2 = 2 * torch.rand(hidden_sizes[1]) - 1

        self.W3 = 2 * torch.rand([output_size, hidden_sizes[1]]) - 1
        self.b3 = 2 * torch.rand(output_size) - 1


    def stable_softmax(z):

        exps = torch.exp(z - torch.max(z))
        return exps / torch.sum(exps)
    
    def feedforward(self, x):

        z1 = torch.matmul(self.W1, x) + self.b1
        a1 = torch.sigmoid(z1)

        z2 = torch.matmul(self.W2, a1) + self.b2
        a2 = torch.sigmoid(z2)

        z3 = torch.matmul(self.W3, a2) + self.b3
        yhat = self.stable_softmax(z3)

        return yhat

    
    def backprop(self, x, y):
        """
        - x is a single image (vector of length 784)
        - y is one-hot-encoding of true digit (vector of length 10)
        """

        z1 = torch.matmul(self.W1, x) + self.b1
        a1 = torch.sigmoid(z1)

        z2 = torch.matmul(self.W2, a1) + self.b2
        a2 = torch.sigmoid(z2)

        z3 = torch.matmul(self.W3, a2) + self.b3
        yhat = self.stable_softmax(z3)

        loss = -torch.sum(y * torch.log(yhat))

        grad_b3 = yhat - y
        print(grad_b3.shape)
        print(a2.T.shape)
        grad_W3 = torch.outer(grad_b3, a2)
        grad_b2 =  torch.sigmoid(torch.matmul(torch.t(self.W3), grad_b3)) * (1 - torch.sigmoid(torch.matmul(torch.t(self.W3), grad_b3)))
        grad_W2 = torch.outer(grad_b2, a1)
        grad_b1 = torch.sigmoid(torch.matmul(torch.t(self.W2), grad_b2)) * (1 - torch.sigmoid(torch.matmul(torch.t(self.W2), grad_b2)))
        grad_W1 = torch.outer(grad_b1, x)

        bias_grads = [grad_b1, grad_b2, grad_b3]
        weight_grads = [grad_W1, grad_W2, grad_W3]

        return {"bias": bias_grads, "weight" : weight_grads, "loss": loss}
    

    def batch_descent(self, images, labels, lr):
        """
        Performs a step of gradient descent from a batch of training data.
        - images are images of the batch
        - labels are the lables of the images in the batch
        - lr is the learning rate
        """

        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        batch_W1_grad = torch.zeros(self.W1.shape)
        batch_b1_grad = torch.zeros(self.b1.shape)
        batch_W2_grad = torch.zeros(self.W2.shape)
        batch_b2_grad = torch.zeros(self.b2.shape)
        batch_W3_grad = torch.zeros(self.W3.shape)
        batch_b3_grad = torch.zeros(self.b3.shape)

        batch_loss = 0
        for (image, label) in zip(images, labels):
            
            #One hot encode label
            label_vec = torch.zeros(output_size)
            label_vec[label] = 1

            single_grads = self.backprop(image, label_vec)
            batch_loss += single_grads["loss"]

            batch_W1_grad = batch_W1_grad + single_grads["weight"][0]
            batch_W2_grad = batch_W2_grad + single_grads["weight"][1]
            batch_W3_grad = batch_W3_grad + single_grads["weight"][2]

            batch_b1_grad = batch_b1_grad + single_grads["bias"][0]
            batch_b2_grad = batch_b2_grad + single_grads["bias"][1]
            batch_b3_grad = batch_b3_grad + single_grads["bias"][2]

        d = len(images)
        self.W1 = self.W1 - lr/d * batch_W1_grad
        self.W2 = self.W2 - lr/d * batch_W2_grad
        self.W3 = self.W3 - lr/d * batch_W3_grad

        self.b1 = self.b1 - lr/d * batch_b1_grad
        self.b2 = self.b2 - lr/d * batch_b2_grad
        self.b3 = self.b3 - lr/d * batch_b3_grad

        return batch_loss
    
    def train(self, epochs, trainloader, lr):

        time0 = time()
        losses = []
        for e in range(epochs):
            
            running_loss = 0
            
            for images, labels in trainloader:

                batch_loss = self.batch_descent(images, labels, lr)
                running_loss += batch_loss
            
            else: 
                print(f"Epoch {e} - Training loss: {running_loss/len(trainloader)}")
                losses.append(running_loss/len(trainloader))

        print(f"Training Time (in minutes) = {(time()-time0)/60}")
        return losses


##Build model with pytorch
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.Sigmoid(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.Sigmoid(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))


#Initialize weights of 0 to every layer
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.)
        m.bias.data.fill_(0.)

init_zero = False
if init_zero:
    model.apply(init_weights)


criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1)


##Fit model
time0 = time()
epochs = 20
losses = []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Forward pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Epoch {e} - Training loss: {running_loss/len(trainloader)}")
        losses.append(running_loss/len(trainloader))

print(f"Training Time (in minutes) = {(time()-time0)/60}")

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Training loss")
if init_zero:
    plt.savefig("learning_curve_pytorch_zeroinit.png")
else:
    plt.savefig("learning_curve_pytorch.png")
plt.show()

## Evaluation
correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print(f"Model Accuracy = {(correct_count/all_count)}")




