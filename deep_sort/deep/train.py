import argparse
import os
import time
import natsort

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset
from PIL import Image

from model import Net

# parser = argparse.ArgumentParser(description="Train on market1501")
# parser.add_argument("--data-dir",default='data',type=str)
# parser.add_argument("--no-cuda",action="store_true")
# parser.add_argument("--gpu-id",default=0,type=int)
# parser.add_argument("--lr",default=0.1, type=float)
# parser.add_argument("--interval",'-i',default=20,type=int)
# parser.add_argument('--resume', '-r',action='store_true')
# args = parser.parse_args()

# TODO: Adathalmaz létrehozása a mappákban: train test split
# Q: Hogyan menjen a tesztelés, test mappában külön 
# Q: Random crop mire jó
# Q: Előről betanítani mennyire tré?

class ISSIADataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        
        self.classes = set([int(imgname[:-4].split('_')[1]) for imgname in self.total_imgs])
        self.classes = sorted(self.classes)
        self.classes_to_idx = {cID : idx for idx, cID in enumerate(self.classes)}

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        label = int(self.total_imgs[idx][:-4].split('_')[1])
        frameNum = int(self.total_imgs[idx][:-4].split('_')[0])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        label = self.classes_to_idx[label]
        return tensor_image, label, frameNum


# Global variables
data_dir = '/home/dobreff/work/Dipterv/MLSA20/data/CNN_dataset/finetune/'
no_cuda = True
gpu_id = 0 
lr = 0.1
interval = 5
resume = False
new_checkpoint_path = './checkpoint/ckpt_new.t7'



# device
device = "cuda:{}".format(gpu_id) if torch.cuda.is_available() and not no_cuda else "cpu"
if torch.cuda.is_available() and not no_cuda:
    cudnn.benchmark = True

# data loading
root = data_dir
train_dir = os.path.join(root,"train")
test_dir = os.path.join(root,"test")
transform_train = torchvision.transforms.Compose([
    #torchvision.transforms.RandomCrop((128,64),padding=4),
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# TODO: Dataset ne image folder legyen, hanem ISSIADataset
trainloader = torch.utils.data.DataLoader(
    ISSIADataSet(train_dir, transform=transform_train),
    batch_size=64,shuffle=True
)
# TODO: Dataset ne image folder legyen, hanem ISSIADataset
testloader = torch.utils.data.DataLoader(
    ISSIADataSet(test_dir, transform=transform_test),
    batch_size=64,shuffle=True
)
num_classes = max(len(trainloader.dataset.classes), len(testloader.dataset.classes))
#num_classes = 751
# net definition
start_epoch = 0
net = Net(num_classes=num_classes)
if resume:
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    checkpoint = torch.load("./checkpoint/ckpt.t7", map_location=torch.device('cpu'))
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
net.to(device)

# loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.0

# train function for each epoch
def train(epoch):
    print("\nEpoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.0
    train_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    for idx, (inputs, labels, frameNum) in enumerate(trainloader):
        # forward
        inputs,labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print 
        if (idx+1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss / interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.0
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels, frameNum) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to a")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, new_checkpoint_path)

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(start_epoch, start_epoch+40):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%20==0:
            lr_decay()


if __name__ == '__main__':
    main()