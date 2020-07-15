import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset

from PIL import Image

import argparse
import os
import natsort

from model import Net

# Define my custom dataset which contains the ISSIA Dataset

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
        return tensor_image, label, frameNum

rootPath = os.path.dirname(os.path.abspath(__file__))

# Global variables
checkpoint_path = os.path.join(rootPath, "checkpoint/ckpt_new.t7")
gpu_id = 0
no_cuda = True
data_dir = '/home/dobreff/work/Dipterv/MLSA20/data/CNN_dataset/vendeg_elorol/'
print(checkpoint_path)

# device
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loader
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataloader = torch.utils.data.DataLoader(
    ISSIADataSet(data_dir, transform=transform),
    batch_size=64, shuffle=False
)

# net definition
net = Net(reid=True, num_classes=21)
assert os.path.isfile(checkpoint_path), "Error: no checkpoint file found!"
print('Loading from {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict)
net.eval()
net.to(device)

# compute features
query_features = torch.tensor([]).float()
query_labels = torch.tensor([]).long()
query_framenums = torch.tensor([]).long()


with torch.no_grad():
    for idx,(inputs, labels, frameNum) in enumerate(dataloader):
        inputs = inputs.to(device)
        features = net(inputs).cpu()
        query_features = torch.cat((query_features, features), dim=0)
        query_labels = torch.cat((query_labels, labels))
        query_framenums = torch.cat((query_framenums, frameNum))
# save features
features = {
    "qf": query_features,
    "ql": query_labels,
    "qframe": query_framenums,
}
torch.save(features,os.path.join(rootPath, "features2.pth"))