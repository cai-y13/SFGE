import torch
import json

def load_keys(key_file):
    with open(key_file, 'r') as f:
        keys = json.load(f)
    return keys

def sparse_to_dense(grad, shape):
    dense_grad = torch.zeros(shape)
    if grad[0].device.type == 'cuda':
        dense_grad = dense_grad.cuda()
    dense_grad = dense_grad.put_(grad[1], grad[0])
    return dense_grad

def cryptography(model, key_file, cfg, op_type='encrypt'):
    keys = load_keys(key_file)
    intensity = cfg.ENCRYPT.INTENSITY
    N = cfg.ENCRYPT.NUM
    for _, (name, param) in enumerate(model.named_parameters()):
        if name not in keys:
            continue
        shape = param.data.shape
        sign = torch.tensor(keys[name]['sign'])
        sign = sign[:min(sign.shape[0], N)]
        if op_type == 'encrypt':
            sign *= intensity
        elif op_type == 'decrypt':
            sign *= -intensity
        index = torch.tensor(keys[name]['index'])
        index = index[:min(index.shape[0], N)]
        if param.device.type == 'cuda':
            sign, index = sign.cuda(), index.cuda()
        grad = [sign, index]
        layer_grad = sparse_to_dense(grad, shape)
        param.data += layer_grad
    return model
