import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from resnet import PreResNet18, ResNet50, ResNet34, ResNet32


def generalized_resnet50(num_classes, args, ssl=False):
    model = ResNet50()
    if ssl == True:
        sd = torch.load(args.ssl_path)['model']
        sd = {k.removeprefix('encoder.module.') : v for k,v in sd.items()}
        model.load_state_dict(sd, strict=False)
    model.fc = nn.Linear(args.embedding_dim, num_classes)
    return model
def generalized_resnet32(num_classes, args, ssl=False):
    """
    ResNet32 cho CIFAR với option load SSL pretrain từ args.ssl_path.
    """
    model = ResNet32(num_classes=num_classes)

    if ssl and getattr(args, "ssl_path", None):
        sd = torch.load(args.ssl_path, map_location="cpu")["model"]
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("encoder.module."):
                new_key = k[len("encoder.module."):]
            elif k.startswith("encoder."):
                new_key = k[len("encoder."):]
            else:
                new_key = k
            cleaned[new_key] = v

        model.load_state_dict(cleaned, strict=False)

    return model


def generalized_resnet50_clothing(num_classes, args):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(args.embedding_dim, num_classes)
    return model

def generalized_preresnet18(num_classes, args, ssl=False):
    model = PreResNet18(num_classes)
    if ssl == True:
        sd = torch.load(args.ssl_path)['model']
        sd = {k.removeprefix('encoder.module.') : v for k,v in sd.items()}
        model.load_state_dict(sd, strict=False)
    return model

def generalized_resnet34(num_classes, args, ssl=False):
    model = ResNet34(num_classes)
    if ssl == True:
        sd = torch.load(args.ssl_path)['model']
        sd = {k.removeprefix('encoder.module.') : v for k,v in sd.items()}
        model.load_state_dict(sd, strict=False)
    return model