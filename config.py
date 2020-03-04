from easydict import EasyDict as edict

class Configuration():

    # Model definition
    MODEL = edict()
    MODEL.NAME = 'resnet18'
    
    # Dataset
    SAMPLE_DATASET = edict()
    SAMPLE_DATASET.DIR = '/home/cai-y13/imagenet/val/'
    SAMPLE_DATASET.MEAN = [0.485, 0.456, 0.406]
    SAMPLE_DATASET.STD = [0.229, 0.224, 0.225]

    VAL_DATASET = edict()
    VAL_DATASET.DIR = '/home/cai-y13/imagenet/val/'
    VAL_DATASET.MEAN = [0.485, 0.456, 0.406]
    VAL_DATASET.STD = [0.229, 0.224, 0.225]

    # Encrypt
    ENCRYPT = edict()
    ENCRYPT.NUM = 20
    ENCRYPT.INTENSITY = 0.2
    ENCRYPT.INCREMENTAL = False
    ENCRYPT.MAX_PERCENT = 0.001

    # Device
    DEVICE = edict()
    DEVICE.CUDA = True

cfg = Configuration()
