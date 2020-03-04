import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from config import cfg
from utils.encrypt_model import cryptography

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

normalize = transforms.Normalize(mean=cfg.VAL_DATASET.MEAN, std=cfg.VAL_DATASET.STD)

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(cfg.VAL_DATASET.DIR, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=10, shuffle=False,
        num_workers=4, pin_memory=True)

key_file = 'keys.json'

model = cryptography(model, key_file, cfg, 'encrypt')

torch.save(model.state_dict(), 'encrypted_model.pth')
# evaluate the encrypted model
total = 0
correct_1 = 0
correct_5 = 0
model.eval()
for data in tqdm(val_loader):
    images, labels = data
    if cfg.DEVICE.CUDA:
        images = images.cuda()
        labels = labels.cuda()
    outputs = model(images)
    _, predict = outputs.topk(5, 1, True, True)
    predict = predict.t()
    correct = predict.eq(labels.view(1, -1).expand_as(predict))
    correct_1 += correct[:1].view(-1).float().sum(0, keepdim=True)
    correct_5 += correct[:5].view(-1).float().sum(0, keepdim=True)
    total += labels.size(0)

print('Top-1 Accuracy: %.3f%%' % (100. * float(correct_1) / float(total)))
print('Top-5 Accuracy: %.3f%%' % (100. * float(correct_5) / float(total)))
'''
model = cryptography(model, key_file, cfg, 'decrypt')

total = 0
correct_1 = 0
correct_5 = 0
model.eval()
for data in tqdm(val_loader):
    images, labels = data
    if cfg.DEVICE.CUDA:
        images = images.cuda()
        labels = labels.cuda()
    outputs = model(images)
    _, predict = outputs.topk(5, 1, True, True)
    predict = predict.t()
    correct = predict.eq(labels.view(1, -1).expand_as(predict))
    correct_1 += correct[:1].view(-1).float().sum(0, keepdim=True)
    correct_5 += correct[:5].view(-1).float().sum(0, keepdim=True)
    total += labels.size(0)

print('Top-1 Accuracy: %.3f%%' % (100. * float(correct_1) / float(total)))
print('Top-5 Accuracy: %.3f%%' % (100. * float(correct_5) / float(total)))
'''
