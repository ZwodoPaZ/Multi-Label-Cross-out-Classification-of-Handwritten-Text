import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, batch_size, device, train_size, val_size, filename, binary = False):
    # For each epoch
    # tensorboard_logging_path = insert path here
    # writer = SummaryWriter(log_dir=tensorboard_logging_path)
    best_validation_loss = float('inf')
    best_training_loss = float('inf')
    best_training_accuracy = 0
    best_validation_accuracy = 0
    for i in range(num_epochs):
        
        model.train()
        total_validation_loss = 0
        total_training_loss = 0 
        correct_train = 0
        correct_validation = 0
        for batch_nr, (images, labels, index) in enumerate(train_loader):

            if binary:
                labels = F.one_hot(labels, num_classes=2).float()
            
            images = images.to(device)
            labels = labels.to(device)
            
            prediction = model(images)
            
            training_loss = criterion(prediction, labels) 
            
            total_training_loss += training_loss.item()
            
            training_loss.backward()
            #writer.add_scalar('Loss Batch / Train', training_loss, i*batch_size + batch_nr)
            optimizer.step()
            optimizer.zero_grad()
            
            print(
                '\rEpoch {} [{}/{}] - Loss: {}'.format(
                    i+1, batch_nr+1, len(train_loader), training_loss
                ),
                end=''
            )
            
            if binary:
                pred_classes = (torch.sigmoid(prediction) > 0.5).float()
                correct_train += (pred_classes == labels).all(dim=1).sum().item()
            else:
                correct_train += torch.sum((torch.max(F.softmax(prediction, dim=1),dim=1)[1]==labels)).item()
        #writer.add_scalar('Loss Epoch / Train', total_training_loss/train_size, i)
        
        model.eval()
        with torch.no_grad():
            for validation_nr, (images, labels, index) in enumerate(val_loader):
                
                if binary:
                    labels = F.one_hot(labels, num_classes=2).float()

                images = images.to(device)
                labels = labels.to(device)
                
                prediction = model(images)
                
                validation_loss = criterion(prediction, labels)
                
                #writer.add_scalar('Loss Batch / Validation', validation_loss, i*batch_size + validation_nr)
                total_validation_loss += validation_loss.item()
                
                if binary:
                    pred_classes = (torch.sigmoid(prediction) > 0.5).float()
                    correct_validation += (pred_classes == labels).all(dim=1).sum().item()
                else:
                    correct_validation += torch.sum((torch.max(F.softmax(prediction, dim=1),dim=1)[1]==labels)).item()
                    
            if total_validation_loss < best_validation_loss:
                best_validation_loss = total_validation_loss
                torch.save(model, filename)
        scheduler.step(total_validation_loss)
        del images, labels, prediction, validation_loss, training_loss
        #writer.add_scalar("Loss Epoch / Validation", total_validation_loss/val_size, i)
        #writer.flush()
    
    #writer.close()
    return model