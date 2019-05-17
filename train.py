'''

Author: nyismaw
Trainer for Casava image files as a part of an in-class kaggle competition
https://www.kaggle.com/c/cassava-disease/

Adopted from the official PyTorch TRAINING A CLASSIFIER tutorial
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import data_loader
import constants
import model
import time
import sys
import constants as C

def train(data_loader, model, train=True):
    print("Training started ... ")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=C.LEARNING_RATE
                                                            , weight_decay=C.WEIGHT_DECAY)
    model.train()
    for epoch in range(C.EPOCH):
        print("Epoch %d started"%(epoch))
        running_loss = 0.0
        start_time   = time.time()
        for batch_index, (train, label) in enumerate(data_loader):

            optimizer.zero_grad()  # Reset the gradients
            prediction = model(train)  # Feed forward
            loss = criterion(prediction, label.long())  # Compute losses
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_index % C.PRINT_TRAIN_INFO_INTERVAL == C.PRINT_TRAIN_INFO_INTERVAL -1:
                    print('[%d, %5d/%5d] loss: %.3f  . Time %5dsec' %
                          (epoch, batch_index, data_loader.__len__()
                          , running_loss / C.PRINT_TRAIN_INFO_INTERVAL
                          , time.time() - start_time))
                    running_loss = 0.0

            if batch_index % C.SAVE_MODEL_INTERVAL == C.SAVE_MODEL_INTERVAL -1:
                    print('Saving model')
                    torch.save(model.state_dict(), "models/CassavaImagesDataset-"
                                                +str(epoch)+"-"+str(batch_index)
                                                +".pt")
    print("Training ended ... ")
