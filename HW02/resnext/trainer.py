
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, trainset, testset):
        #self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        #                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=False, num_workers=2)

        #self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        #                                       download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)

    
    def compute_loss_accuracy(self, dataloader, net, criterion):
        count = 0
        full_loss = 0
        
        correct_predictions = 0
        all_predictions = 0
        with torch.no_grad():
            for data in dataloader:
                count += 1
                inputs, labels = data

                outputs = net(inputs)

                loss = criterion(outputs, labels)
                full_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                # print(predicted, labels, predicted == labels, torch.sum(predicted == labels))
                
                all_predictions += labels.shape[0]
                correct_predictions += torch.sum(predicted == labels).item()
                
        full_loss /= count
        
        return (full_loss, correct_predictions / all_predictions)
    
    def train(self, net, should_log=True):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        
        sw = SummaryWriter('logs')
        
        for epoch in range(1):  # loop over the dataset multiple times

            #running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                inputs, labels = data
                print('inputs in train: ', inputs.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                #running_loss += loss.item()
                if should_log:
                    
                    loss, acc = self.compute_loss_accuracy(self.trainloader, net, criterion)
                    sw.add_scalar('trainloss', loss)
                    
                    loss, acc = self.compute_loss_accuracy(self.testloader, net, criterion)
                    
                    sw.add_scalar('testloss', loss)
