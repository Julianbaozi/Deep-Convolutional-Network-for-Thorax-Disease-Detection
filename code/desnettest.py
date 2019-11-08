import sys
sys.path.append("../")
from model1 import *
from model1 import BasicCNN,myresnet,mymodel
from resnet import resnet18
from mybceloss import MyBCEloss,new_BCEloss,BCEloss_batch

import numpy as np
import PIL
from metrics import evaluate
import pickle
import torchvision.models as models

from xray_dataloader_z_score import ChestXrayDataset, create_split_loaders
import torchvision
from torchvision import transforms, utils
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
# Setup: initialize the hyperparameters/variables
num_epochs = 20         # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.0005  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

def test(model,dataset,computing_device):
    with torch.no_grad():
        temp_loss = 0
        temp_acc = 0
        temp_precision=0
        temp_recall = 0
        temp_BCR =0
        temp_acc_list = np.zeros(14)
        temp_precision_list=np.zeros(14)
        temp_recall_list = np.zeros(14)
        temp_BCR_list =np.zeros(14)
        for minibatch_count,(images,labels) in enumerate(dataset,0):
            images,labels = images.to(computing_device),labels.to(computing_device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            (acc,pre,rec,BCR),(acc_list,pre_list,rec_list,BCR_list) = evaluate(outputs.cpu().data.numpy(),\
                                                                               label = labels.cpu().data.numpy())
            
            temp_acc_list += acc_list
            temp_precision_list+=pre_list
            temp_recall_list += rec_list
            temp_BCR_list +=BCR_list
            
            temp_loss += loss
            temp_acc +=acc
            temp_precision+=pre
            temp_recall +=rec
            temp_BCR +=BCR
            
        temp_acc_list /= (minibatch_count+1)
        temp_precision_list/= (minibatch_count+1)
        temp_recall_list/= (minibatch_count+1)
        temp_BCR_list /= (minibatch_count+1)
            
        temp_loss= temp_loss/(minibatch_count+1)
        temp_acc= temp_acc/(minibatch_count+1)
        temp_precision= temp_precision/(minibatch_count+1)
        temp_recall= temp_recall/(minibatch_count+1)
        temp_BCR= temp_BCR/(minibatch_count+1)
        print( temp_BCR_list )
        print("loss after %d minibatch is %.3f,acc is %.3f,precision is %.3f,recall is %.3f,BCR is %.3f"%(minibatch_count,temp_loss,temp_acc,temp_precision,temp_recall,temp_BCR))
        return(temp_loss,(temp_acc,temp_precision,temp_recall,temp_BCR),(temp_acc_list,temp_precision_list,temp_recall_list,temp_BCR_list))



# Instantiate a BasicCNN to run on the GPU or CPU based on CUDA support
# model =BasicCNN()
# model = ResNet18()
# model = models.resnet18(num_classes=14)
# model = resnet18(num_classes=14)
  #TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
                                # transforms.Resize((512,512)),
                                # transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                                # transforms.ColorJitter(brightness=64/255, contrast=.25, saturation=.25, hue=.04), 
                                #http://www.voidcn.com/article/p-dmjhonsq-bgn.html
#                                 
                                # because scale doesn't always give 224 x 224, this ensures 224 x
                                # 224
                                transforms.CenterCrop(224),
                                transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor()])

# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 4, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)
# (pn_weight,class_weight)=get_data_weight(train_loader)
# class_weight = torch.tensor(class_weight).float().to(computing_device)
#TODO: Define the loss criterion and instantiate the gradient descent optimizer
# criterion = torch.nn.BCELoss()
# criterion=MyBCEloss(computing_device,True,True)
# criterion= new_BCEloss(computing_device)
criterion= BCEloss_batch(computing_device)
#TODO: Instantiate the gradient descent optimizer - use Adam optimizer with default parameters

# optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum = 0.9, weight_decay = 1e-4)
# print(model)

def trainer(keep_training = False):

    model_path = '../save/densenet_result.ckpt'
    save_data_path = '../save/densenet_result.pkl'
    
    model = torchvision.models.densenet161(pretrained='imagenet')
    n_class = 14
    freeze = True
    if freeze:
        for i, param in model.named_parameters():
            param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, n_class)    
    model = model.to(computing_device)
    
    if keep_training:
        model.load_state_dict(torch.load(model_path))
        picklefile = open(save_data_path, 'rb')
        (train_loss,validation_loss) = pickle.load(picklefile)
        picklefile.close()
    else: 
        validation_loss = []
        train_loss=[]

    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # trainer()
    # Track the loss across training
    total_loss = []
    avg_minibatch_loss = []
    # picklefile = open('save/myresnet_result.pkl', 'rb')
    # (train_loss,validation_loss) = pickle.load(picklefile)
    # print(train_loss,val_loss)
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate) #TODO - optimizers are defined in the torch.optim package
    # Begin training procedure
    for epoch in range(num_epochs):
        N = 300
        N_minibatch_loss = 0.0  
        N_acc=0.0
        N_minibatch_acc = 0.0 
        N_minibatch_precision = 0.0
        N_minibatch_recall = 0.0 
        N_minibatch_BCR = 0.0
        temp_train_loss = 0.0
        # Get the next minibatch of images, labels for training
        for minibatch_count, (images, labels) in enumerate(train_loader, 0):
            # Put the minibatch data in CUDA Tensors and run on the GPU if supported
            images, labels = images.to(computing_device), labels.to(computing_device)
    #         print(images[0])
            #print(images.requires_grad)
            # Zero out the stored gradient (buffer) from the previous iteration
            optimizer.zero_grad()
            
            # Perform the forward pass through the network and compute the loss
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs,labels)

            temp_train_loss+=  loss
    #         print(outputs)
    #         assert 0==1
    #         print(output.shape)
    #         outputs = torch.sigmoid(outputs)
            output =outputs.cpu().data.numpy()
            label = labels.cpu().data.numpy()
    #         print(label.shape)
            (accuracy,precision,recall,BCR),_ = evaluate(output,label)
    #         print(outputs,labels)

    #         print(loss)
    #         assert 0==1
    #         print(loss)
    #         print(loss.shape)
    #         print(loss)
    #         assert 0==1
            # Automagically compute the gradients and backpropagate the loss through the network
            loss.backward()

            # Update the weights
            optimizer.step()

            # Add this iteration's loss to the total_loss
            total_loss.append(loss.item())
            N_minibatch_loss += loss
            N_minibatch_acc += accuracy
            #TODO: Implement validation
            if minibatch_count % N == 0 and minibatch_count!=0:    
                # Print the loss averaged over the last N mini-batches   
                N_minibatch_loss /= N
                N_minibatch_acc /= N
                print('Epoch %d, average minibatch %d loss: %.3f acc:%.3f'%(epoch + 1, minibatch_count, N_minibatch_loss,N_minibatch_acc ))
                # Add the averaged loss over N minibatches and reset the counter
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0
                N_minibatch_acc =0.0
                N_acc=0.0
    #             if minibatch_count %600 ==0 and  minibatch_count!=0:

        temp_train_loss /=  minibatch_count
        train_loss.append(temp_train_loss)
        print(epoch+1, "epoch training complete training loss is %.3f"%temp_train_loss)

        print('here we do validation')
        temp_loss,_,_ = test(model,val_loader,computing_device)
        validation_loss.append(temp_loss)
        #TODO early stopping
        if len(validation_loss)>1  and  (validation_loss[-1] >validation_loss[-2] ):
            print('here we use early stopping')
            break
        else:
            torch.save(model.state_dict(), model_path)
        print("Finished", epoch + 1, "epochs of training") 

    print("Training complete after", epoch+1, "epochs")
    print("Here we do test")
    # model = resnet18(num_classes=14)
    # if modelname == 'myresnet':
    #     testmodel = myresnet()
    # if modelname == 'mymodel':
    #     testmodel = mymodel()
    testmodel = torchvision.models.densenet161(pretrained='imagenet')
    testmodel = model.to(computing_device)
    testmodel.load_state_dict(torch.load(model_path))
    test_loss,average_score,percalss_score=test(testmodel,test_loader,computing_device)
    output = open(save_data_path, 'wb')
    pickle.dump((train_loss,validation_loss), output)
    pickle.dump(average_score, output)
    pickle.dump(percalss_score, output)
    output.close()

if __name__ == "__main__":
    trainer()
    