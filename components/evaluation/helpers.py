# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data, transform_torch_to_cv2, inv_normalize
import components.torchvision_utilities.transforms as T
import components.torchvision_utilities.utils as utils
from components.evaluation.utils_evaluator import LogHelpers
from components.neural_nets.NNClassifier import ChooseModel
from matplotlib import cm
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


def plot_w_bb(im, target, target_pred, targets_success, predict_success, inv_norm=False, plot_boxes=False):

    # print(target)
    # print(target_pred)

    # im = np.reshape(im, (224, 224, 3))

    if inv_norm:
        im = inv_normalize(im)

    im = transform_torch_to_cv2(im)

    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    boxes = target["boxes"].numpy().astype(np.uint32)
    boxes_pred = target_pred["boxes"].numpy().astype(np.uint32)

    masks = target["masks"].numpy()
    masks_pred = target["masks"].numpy()
    print(np.shape(masks_pred))

    if plot_boxes:
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
                    xc = (xc + 25).astype(np.uint64)
                yc = ((boxes[i][3] / 2 + boxes[i][1] / 2)).astype(np.uint64)

                try:
                    cv2.putText(
                        im, str(target["labels"][i].numpy()
                                ), (xc, yc), 1, 0.8, (255, 0, 0), 1
                    )
                except:
                    print("Cannot print labels")
    else:
        if len(masks[0]) > 0:
            for i, mask in enumerate(masks):

                xc = boxes_pred[i][2] / 2 + boxes_pred[i][0] / 2
                if np.abs(xc) > np.abs(xc - im.shape[0]):
                    xc = (xc - 50).astype(np.uint64)
                else:
                    xc = (xc + 25).astype(np.uint64)
                yc = ((boxes_pred[i][3] / 2 + boxes_pred[i]
                       [1] / 2)).astype(np.uint64)

                try:
                    cv2.putText(
                        im, str(target["labels"][i].numpy()
                                ), (xc, yc), 1, 0.8, (255, 0, 0), 1
                    )
                except:
                    print("Cannot print labels")

                cnts = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts:
                    cv2.drawContours(im, [c], -1, (255, 0, 0), thickness=2)

    # Show image
    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    image = copy.copy(im)
    if plot_boxes:
        if len(boxes_pred) == 0:
            print("No predictions")
            return
    else:
        if len(masks_pred) == 0:
            print("No predictions")
            return

    if plot_boxes:
        if len(boxes_pred[0]) > 0:
            for i in range(len(predict_success)):
                if i < len(predict_success):
                    print(predict_success)
                    if predict_success[i]:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                else:
                    color = (0, 0, 255)

                # im = copy.copy(image)
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
                    xc = (xc + 25).astype(np.uint64)
                yc = ((boxes_pred[i][3] / 2 + boxes_pred[i]
                       [1] / 2)).astype(np.uint64)

                try:
                    cv2.putText(
                        im, str(target_pred["labels"][i].numpy()
                                ), (xc, yc), 1, 0.8, color, 1
                    )
                except:
                    print("Cannot print labels")

                if i > 7:
                    break

                print(f'Score: {target_pred["scores"][i].numpy()}')
                # Show image
            cv2.imshow("Image", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        if len(masks_pred[0]) > 0:
            for i, mask_pred in enumerate(masks_pred):
                if i < len(predict_success):
                    print(predict_success)
                    if predict_success[i]:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                else:
                    color = (0, 0, 255)

                xc = boxes_pred[i][2] / 2 + boxes_pred[i][0] / 2
                if np.abs(xc) > np.abs(xc - im.shape[0]):
                    xc = (xc - 50).astype(np.uint64)
                else:
                    xc = (xc + 25).astype(np.uint64)
                yc = ((boxes_pred[i][3] / 2 + boxes_pred[i]
                       [1] / 2)).astype(np.uint64)

                try:
                    cv2.putText(
                        im, str(target_pred["labels"][i].numpy()
                                ), (xc, yc), 1, 0.8, color, 1
                    )
                except:
                    print("Cannot print labels")

                overlay_pred = np.zeros(im.shape, im.dtype)
                overlay_pred[:, :] = (0, 255, 0)
                print(np.shape(mask_pred))
                print(np.shape(im))
                mask_pred_copy = cv2.bitwise_and(
                    overlay_pred, overlay_pred, mask=mask_pred)
                im = cv2.addWeighted(mask_pred_copy, 0.2, im, 0.8, 0)

            cv2.imshow("Image", im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


@torch.no_grad()
def evaluate(model, data_loader_test, device, show_plot=True, inv_norm=True):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # Metric logger class
    logger = LogHelpers(binary=False)

    # Put in evaluation mode
    model.eval()

    Tpos_vec = 0
    Fpos_vec = 0
    Fneg_vec = 0
    Tneg_vec = 0
    accuracy_vec = []

    success_vec = []

    pics = 0

    data_iter_test = iter(data_loader_test)
    # iterate over test subjects
    for images, targets in data_iter_test:
        billeder = images
        images = list(img.to(device) for img in images)

        # torch.cuda.synchronize()  # what is this??
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        for i, prediction in enumerate(outputs):
            pics = pics + 1
            logger.__load__(targets[i], prediction)
            label, score = logger.get_highest_predictions(score_limit=0.5)
            data, targets_success, predict_success = logger.calc_accuracy(
                score, overlap_limit=0.5)
            Tpos, Fpos, Fneg, Tneg = data

            accuracy = (Tpos+Tneg)/(Tpos+Tneg+Fpos+Fneg)

            accuracy_vec.append(accuracy)
            Tpos_vec += Tpos
            Fpos_vec += Fpos
            Fneg_vec += Fneg
            Tneg_vec += Tneg

            if targets_success is not None:
                for val in targets_success:
                    if val != 0:
                        success_vec.append(1)
                    else:
                        success_vec.append(0)

            if show_plot:
                # print(f'Labels success: {label}')
                # print(f'Targets success: {targets_success}')
                if targets_success is not None:
                    plot_w_bb(billeder[i], targets[i], prediction,
                              targets_success, predict_success, inv_norm)
                    # show_plot = True

    success_percent = success_vec.count(1) / len(success_vec)

    # data = sum(accuracy_vec) / len(accuracy_vec)

    torch.set_num_threads(n_threads)

    # Set back in training mode
    model.train()

    return sum(accuracy_vec)/len(accuracy_vec), success_percent, Tpos_vec, Fpos_vec, Fneg_vec, Tneg_vec


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
