from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn as nn
import numpy as np
import torchvision
import torch


def getFastRCNNResnet50Fpn(pretrained):
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)


def getMaskRCNNResnet50Fpn(pretrained):
    return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)


def conv_out_features(model, shape):
    o = model.cls_score(torch.zeros(1, *shape))
    return int(np.prod(o.size()))


def ChooseModel(input, n_classes, freeze=False):

    if input == "faster":
        print(f"model is Faster RCNN resnet 50")
        model = getFastRCNNResnet50Fpn(pretrained=True)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes)

    elif input == "mask":
        print(f"model is Mask RCNN")
        model = getMaskRCNNResnet50Fpn(pretrained=True)
        if freeze:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           n_classes)
    return model


def main():
    model = ChooseModel("faster", 5, True)
    print(f"model is: {model}")
    print("Pretrained classifiers")


if __name__ == "__main__":
    main()
