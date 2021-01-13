from torchvision import models
import torch


def getWideResnet101(pretrained=True):
    return models.wide_resnet101_2(pretrained=True)


def getAlexNet(pretrained=True):
    return models.alexnet(pretrained=True)


def ChooseModel(n_classes, pretrained):

    model = getWideResnet101(pretrained=pretrained)
    in_features = model.fc.in_features
    classifier = torch.nn.Linear(in_features, n_classes)
    model.fc = classifier

    return model


if __name__ == "__main__":
    model = ChooseModel(n_classes=2, pretrained=True)
    print(model)
