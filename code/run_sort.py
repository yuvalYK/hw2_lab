import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import shutil

import torchvision
from torchvision import models, transforms, datasets

import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

print("Your working directory is: ", os.getcwd())
torch.manual_seed(0)


# ======================================================
# ======================================================
# ======================================================
# ======================================================

# You are not allowed to change anything in this file.
# This file is meant only for training and saving the model.
# You may use it for basic inspection of the model performance.

# ======================================================
# ======================================================
# ======================================================
# ======================================================

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 0.001

# Paths to your train and val directories
#train_dir = os.path.join("data/data", "train")
#val_dir = os.path.join("data/data", "val")


class extendo(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, index):
        x, y = self.dataset.__getitem__(index)
        
        return x, y, self.dataset.imgs[index][0]
    
    def __len__(self):
        return len(self.dataset)

def imshow(inp, title=None):
    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Resize the samples and transform them into tensors
data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

# Create a pytorch dataset from a directory of images
train_dataset = datasets.ImageFolder('data/data/train', data_transforms)
val_dataset = datasets.ImageFolder('data/data/val', data_transforms)

train_dataset_extend = extendo(train_dataset)

class_names = train_dataset.classes
print("The classes are: ", class_names)

# Dataloaders initialization
train_dataloader = torch.utils.data.DataLoader(train_dataset_extend, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


for i in range(3):
    inputs, classes,_ = next(iter(train_dataloader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

NUM_CLASSES = len(class_names)


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=100):
    """Responsible for running the training and validation phases for the requested model."""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    loss_dict = {'train': [], 'val': []}
    acc_dict = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels,_ in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            acc_dict[phase].append(epoch_acc.item())
            loss_dict[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_dict, acc_dict


# Use a prebuilt pytorch's ResNet50 model
model_ft = models.resnet50(pretrained=False)

# Fit the last layer for our specific task
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=LR)

# Decay LR by a factor of 0.1 every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# Train the model
model_ft, loss_dict, acc_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)

# Save the trained model
torch.save(model_ft.state_dict(), "model_sort_{1}.pt")

softmax = torch.nn.Softmax()

count = 0
max_ = 0


classes = []

os.mkdir('data2')

path = os.path.join('data2', 'train')
os.mkdir(path)

for i in class_names:
    os.mkdir(os.path.join(path, i))

os.mkdir(os.path.join(path, 'unknown'))

for inputs, labels,filenames in tqdm(dataloaders['train']):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    #optimizer.zero_grad()

    # forward
    # track history if only in train
    with torch.set_grad_enabled('train' == 'train'):
        outputs = model_ft(inputs).to('cpu')
        for i in range(len(outputs)):
            prob = softmax(outputs[i])
            argmax = torch.argmax(prob)
            if prob[argmax]>0.7:
                count += 1
                shutil.copy(filenames[i], os.path.join(path, class_names[argmax]))
            else:
                shutil.copy(filenames[i], os.path.join(path, 'unknown'))
                if prob[argmax]>max_:
                    max_=prob[argmax]

        #_, preds = torch.max(outputs, 1)
        #loss = criterion(outputs, labels)
        
print(count)
# Basic visualizations of the model performance
#fig = plt.figure(figsize=(20,10))
#plt.title("Train - Validation Loss")
#plt.plot( loss_dict['train'], label='train')
#plt.plot(  loss_dict['val'], label='validation')
#plt.xlabel('num_epochs', fontsize=12)
#plt.ylabel('loss', fontsize=12)
#plt.legend(loc='best')
#plt.savefig('train_val_loss_plot.png')

#fig = plt.figure(figsize=(20,10))
#plt.title("Train - Validation ACC")
#plt.plot( acc_dict['train'], label='train')
#plt.plot(  acc_dict['val'], label='validation')
#plt.xlabel('num_epochs', fontsize=12)
#plt.ylabel('ACC', fontsize=12)
#plt.legend(loc='best')
#plt.savefig('train_val_acc_plot.png')