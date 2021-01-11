import torchvision
import torch.nn as nn
import torch
import numpy as np


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


def getFastRCNNResnet50Fpn(pretrained):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)


def getMaskRCNNResnet50Fpn(pretrained):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)


def conv_out_features(model, shape):
    o = model.cls_score(torch.zeros(1, *shape))
    return int(np.prod(o.size()))


def get_classifier(out_features, n_classes):
    classifier = nn.Sequential(
        nn.Linear(out_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, n_classes),
        nn.Softmax(dim=1),
    )
    return classifier


def ChooseModel(input, n_classes, freeze=False):

    if input == "faster":
        print(f"model is Faster RCNN resnet 50")
        model = getFastRCNNResnet50Fpn(pretrained=True)
        out_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = get_classifier(
            out_features, n_classes
        )
    elif input == "mask":
        print(f"model is Mask RCNN")
        model = getMaskRCNNResnet50Fpn(pretrained=True)
        out_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor.cls_score = get_classifier(
            out_features, n_classes
        )

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model


def main():
    model = ChooseModel("faster", 5, True)
    print(f"model is: {model}")
    print("Pretrained classifiers")


if __name__ == "__main__":
    main()
