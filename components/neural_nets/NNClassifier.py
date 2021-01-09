import torchvision
import torch.nn as nn


def getCNNFeatureExtractVGG19(pretrained):
    return torchvision.models.vgg19(pretrained=pretrained)


def getCNNFeatureExtractALEX(pretrained):
    return torchvision.models.alexnet(pretrained=True)


def getCNNFeatureExtractRESNET152(pretrained):
    return torchvision.models.resnet152(pretrained=pretrained)


def getCNNFeatureExtractSQUEEZE(pretrained):
    return torchvision.models.squeezenet1_1(pretrained=pretrained)


def getCNNFeatureExtractDENSE(pretrained):
    return torchvision.models.densenet161(pretrained=pretrained)


def getCNNFeatureExtractINCEPTION(pretrained):
    return torchvision.models.inception_v3(pretrained=pretrained)


def getCNNFeatureExtractRESNEXT101(pretrained):
    return torchvision.models.resnext101_32x8d(pretrained=pretrained)


def getCNNFeatureExtractWIDE(pretrained):
    return torchvision.models.wide_resnet101_2(pretrained=True)


def ChooseModel(input, n_classes, freeze=False):

    if input == 'resnet':
        print(f"model is ResNet 152")
        model = getCNNFeatureExtractRESNET152(pretrained=True)

    elif input == 'resnext':
        print(f"model is ResNext-101-32x8d")
        model = getCNNFeatureExtractRESNEXT101(pretrained=True)

    elif input == 'wide':
        print(f"model is Wide ResNet-101-2")
        model = getCNNFeatureExtractWIDE(pretrained=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    out_features = model.fc.in_features

    model.classifier = nn.Sequential(
        nn.Linear(out_features, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(2048, n_classes),
        nn.Softmax(dim=1)
    )

    return model


def main():
    model = ChooseModel('resnet', 4, True)
    print(f"model is: {model}")
    print('Pretrained classifiers')


if __name__ == '__main__':
    main()
