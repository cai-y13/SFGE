import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def get_large_gradients(param, name, number, keys):
    current_grad = param.grad.data.view(-1)
    if name in keys:
        for _, key_index in enumerate(keys[name]['index']):
            current_grad[key_index] = 0
    _, indices = torch.sort(torch.abs(current_grad), descending=True)
    sparse_indices = indices[:number]
    sparse_values = current_grad.take(sparse_indices)
    sparse_values.sign_()
    sparse_grad = [[sparse_values, sparse_indices], param.grad.data.shape, name]
    return sparse_grad

def sparse_to_dense(grad, shape):
    dense_grad = torch.zeros(shape)#.cuda()
    if grad[0].device.type == 'cuda':
        dense_grad = dense_grad.cuda()
    dense_grad = dense_grad.put_(grad[1], grad[0])
    return dense_grad

def updated_gradients(sparse_grad, name):
    updated_grads = dict()
    for grad, shape, name in sparse_grad:
        layer_grad = sparse_to_dense(grad, shape)
        updated_grads[name] = layer_grad
    return updated_grads

def put_grad_to_optimizer(model, updated_grads, intensity):
    for _, (name, param) in enumerate(model.named_parameters()):
        if name in updated_grads:
            param.grad.data = updated_grads[name].clone()

def update_weights(updated_grads, model, intensity):
    for _, (name, param) in enumerate(model.named_parameters()):
        if name not in updated_grads:
            continue
        param.data += intensity * updated_grads[name]
    return model


def key_generation(model, dataloader, criterion, cfg):
    encrypt_num = cfg.ENCRYPT.NUM
    intensity = cfg.ENCRYPT.INTENSITY
    percentage = cfg.ENCRYPT.MAX_PERCENT
    incremental = cfg.ENCRYPT.INCREMENTAL
    encrypt_round = 1 if not incremental else encrypt_num 
    optimizer = optim.SGD(model.parameters(), lr=cfg.ENCRYPT.INTENSITY, \
        momentum=0, weight_decay=0)
    encrypt_per_round = encrypt_num // encrypt_round
    keys = dict()
    for iter in range(encrypt_round):
        model.eval()
        optimizer.zero_grad()
        batch = 0
        for data in tqdm(dataloader):
            images, labels = data
            if cfg.DEVICE.CUDA:
                images = images.cuda()
                labels = labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            batch += 1
            #if batch == 100:
            #    break

        sparse_grads = []
        names = []
        for _, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad:
                if 'bn' in name or 'bias' in name:
                    continue
                number = min(int(len(param.grad.data.view(-1)) * percentage) - \
                        iter * encrypt_per_round, encrypt_per_round) 
                if number <= 0:
                    continue
                large_grad = get_large_gradients(param, name, number, keys)
                sign, index = large_grad[0]
                if name not in keys:
                    keys[name] = dict()
                    keys[name]['sign'] = []
                    keys[name]['index'] = []
                keys[name]['sign'] += sign.cpu().numpy().tolist()
                keys[name]['index'] += index.cpu().numpy().tolist()
                sparse_grads.append(large_grad)
                names.append(name)
        if iter < encrypt_round - 1:
            updated_grads = updated_gradients(sparse_grads, names)
            model = update_weights(updated_grads, model, intensity)
    return keys


