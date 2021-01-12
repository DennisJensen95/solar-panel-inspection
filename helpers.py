# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data
import components.torchvision_utilities.transforms as T
import components.torchvision_utilities.utils as utils
from components.evaluation.utils_evaluator import LogHelpers
from components.neural_nets.NNClassifier import ChooseModel
import torchvision
import pandas as pd
import torch
import json
import copy
import time
import os
import cv2
import numpy as np
import copy


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def plot_w_bb(im, target, target_pred, targets_success):

    print(target)
    print(target_pred)

    im = np.reshape(im, (224, 224, 3))

    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    boxes = target["boxes"].numpy().astype(np.uint32)
    boxes_pred = target_pred["boxes"].numpy().astype(np.uint32)

    if len(boxes[0]) > 0:
        for i in range(len(boxes)):
            cv2.rectangle(
                im,
                (boxes[i][0], boxes[i][1]),
                (boxes[i][2], boxes[i][3]),
                (255, 0, 0),
                2,
            )

            xc = boxes[i][2] / 2 + boxes[i][0] / 2
            if np.abs(xc) > np.abs(xc - im.shape[0]):
                xc = (xc - 50).astype(np.uint64)
            else:
                xc = (xc + 15).astype(np.uint64)
            yc = ((boxes[i][3] / 2 + boxes[i][1] / 2)).astype(np.uint64)

            try:
                cv2.putText(
                    im, str(target["labels"][i].numpy()), (xc, yc), 1, 0.8, (255, 0, 0), 1
                )
            except:
                print("Cannot print labels")
    
    # Show image
    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = copy.copy(im)

    print(boxes_pred)
    if len(boxes_pred[0]) > 0:
        for i in range(len(boxes_pred)):
            if i<len(targets_success):
                if targets_success[i] == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
            else:
                color = (0, 0, 255)
            
            im = copy.copy(image)
            cv2.rectangle(
                im,
                (boxes_pred[i][0], boxes_pred[i][1]),
                (boxes_pred[i][2], boxes_pred[i][3]),
                color,
                2,
            )

            xc = boxes_pred[i][2] / 2 + boxes_pred[i][0] / 2
            if np.abs(xc) > np.abs(xc - im.shape[0]):
                xc = (xc - 50).astype(np.uint64)
            else:
                xc = (xc + 15).astype(np.uint64)
            yc = ((boxes_pred[i][3] / 2 + boxes_pred[i][1] / 2)).astype(np.uint64)

            try:
                cv2.putText(
                    im, str(target_pred["labels"][i].numpy()), (xc, yc), 1, 0.8, color, 1
                )
            except:
                print("Cannot print labels")
            
            print(f'Score: {target_pred["scores"][i].numpy()}')
            # Show image
            cv2.imshow("Image", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


@torch.no_grad()
def evaluate(model, data_loader_test, device, show_plot=True):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # Metric logger class
    logger = LogHelpers()

    # Put in evaluation mode
    model.eval()

    success_array = []
    pics = 0

    data_iter_test = iter(data_loader_test)
    # iterate over test subjects
    for images, targets in data_iter_test:
        billeder = images
        images = list(img.to(device) for img in images)

        # torch.cuda.synchronize()  # what is this??
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for i, image in enumerate(outputs):
            pics = pics + 1
            logger.__load__(image, targets[i])
            label, score = logger.get_highest_predictions(score_limit=0.1)
            success, targets_success, overlaps = logger.get_success_w_box_overlap(
                label, score, overlap_limit=0.1
            )

            if success:
                # print(f'Targets_success: {targets_success}')
                # print(f'Overlaps: {overlaps}')
                for val in targets_success:
                    success_array.append(val)
            else:
                n_targ = len(targets[i]["labels"])
                for k in range(n_targ):
                    success_array.append(0)
            
            if show_plot:
                print(f'Labels success: {label}')
                print(f'Targets success: {targets_success}')
                if targets_success is not None:
                    plot_w_bb(billeder[i].numpy(), targets[i], image, targets_success)
                    show_plot = False


    success_percent = success_array.count(1) / len(success_array)

    # Put back in train mode
    model.train()

    torch.set_num_threads(n_threads)

    return success_percent


def load_configuration(filename):
    if filename == None:
        filename = "model_conf.json"

    with open(filename, "r") as file:
        configuration = json.load(file)

    print(f'Selected model: {configuration["Model"]}')
    print(f'Label configuration: {configuration["Classification"]}')

    return configuration




if __name__ == "__main__":
    print("No default")
