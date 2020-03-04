import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import json
import numpy as np
import random

from config import cfg
from utils.key_generation import key_generation

if cfg.MODEL.NAME == 'vgg16':
    model = models.vgg16(pretrained=True)#.cuda()
elif cfg.MODEL.NAME == 'resnet18':
    model = models.resnet18(pretrained=True)
elif cfg.MODEL.NAME == 'resnet50':
    model = models.resnet50(pretrained=True)
elif cfg.MODEL.NAME == 'resnet101':
    model = models.resnet101(pretrained=True)
else:
    raise Exception('{} is not supported currently, or you can define your own model'.format(cfg.MODEL.NAME))

if cfg.DEVICE.CUDA:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()

normalize = transforms.Normalize(mean=cfg.SAMPLE_DATASET.MEAN, std=cfg.SAMPLE_DATASET.STD)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#setup_seed(20)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(cfg.SAMPLE_DATASET.DIR, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=10, shuffle=True,
        num_workers=20, pin_memory=True)

keys = key_generation(model, val_loader, criterion, cfg)

with open('keys.json', 'w') as f:
    json.dump(keys, f)
