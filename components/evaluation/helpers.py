# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data, transform_torch_to_cv2, inv_normalize
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

def plot_w_bb(im, target, target_pred, targets_success, predict_success, inv_norm=False, plot_boxes=False):

    if inv_norm:
        im = inv_normalize(im)
        
    im = transform_torch_to_cv2(im)

    # Showcase image first
    showcase_image(im)

    # Predictions
    boxes_pred = target_pred["boxes"].numpy().astype(np.uint32)
    masks_pred = target_pred["masks"].numpy().astype(np.uint8)
    
    # Ground truth
    boxes = target["boxes"].numpy().astype(np.uint32)
    masks = target["masks"].numpy()

    # if plot_boxes:
    #     im = plot_type(boxes, im, target["labels"], targets_success, predictions=target_pred["scores"], method="box", color=(255,0,0))
    # else:
    #     im = plot_type(masks, im, target["labels"], targets_success, predictions=target_pred["scores"], method="contours", color=(255,0,0))
        
    # Show image
    # showcase_image(im)

    image = copy.copy(im)
    if len(boxes_pred) == 0:
        print("No predictions")
        return
    
    if plot_boxes:
        im = plot_type(boxes_pred, im, target_pred["labels"], predict_success, predictions=target_pred["scores"], method="box", showcase=True)
    else:
        im = plot_type(masks_pred, im, target_pred["labels"], predict_success, predictions=target_pred["scores"], method="mask", showcase=True)

def plot_type(objects, im, labels, succeses, predictions=None, method="box", showcase=False, color=None):
    # Prediction plots
    get_color_state = True if color == None else False
    if len(objects[0]) > 0:
        for i in range(len(succeses)):
            if get_color_state:
                color = get_color(i, succeses)
            
            text = str(labels[i].numpy())
            im = put_text(objects[i], im, color, text)
            
            if method == "box":
                im = box_plot(im, color, objects[i])
            elif method == "mask":
                im = mask_plot(im, color, objects[i])
            elif method == "contour":
                im = contours_plot(im, objects[i])
                
            if predictions != None and len(predictions) > 0:
                print(f'Score: {predictions[i].numpy()}')
        if showcase:
            showcase_image(im)
    return im
        
def showcase_image(im):
    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def put_text(box, im, color, text):
    xc = box[2] / 2 + box[0] / 2
    if np.abs(xc) > np.abs(xc - im.shape[0]):
        xc = (xc - 50).astype(np.uint64)
    else:
        xc = (xc + 25).astype(np.uint64)
    yc = ((box[3] / 2 + box[1] / 2)).astype(np.uint64)

    try:
        cv2.putText(
            im, text, (xc, yc), 1, 0.8, color, 1
        )
    except:
        print("Cannot print labels")
    return im

def get_color(i, predict_success):
    if i<len(predict_success):
        print(predict_success[i])
        if predict_success[i]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
    else:
        color = (0, 0, 255)
    
    return color
                    
def mask_plot(im, color, mask_pred):
    mask_pred = np.reshape(mask_pred, (mask_pred.shape[1], mask_pred.shape[2], mask_pred.shape[0]))
    overlay_pred = np.zeros(im.shape, im.dtype)
    overlay_pred[:,:] = color
    mask_pred_copy = cv2.bitwise_and(overlay_pred, overlay_pred, mask = mask_pred)
    im = cv2.addWeighted(mask_pred_copy, 1, im, 1, 0, im)
    return im

def box_plot(im, color, box):
    cv2.rectangle(
        im,
        (box[0], box[1]),
        (box[2], box[3]),
        color,
        2,
    )
    return im

def contours_plot(im, mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(im, [c], -1, (255, 0, 0), thickness=2)
    
    return im

@torch.no_grad()
def evaluate(model, data_loader_test, device, show_plot=True, inv_norm = True, score_limit=0.5):
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
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for i, prediction in enumerate(outputs):
            pics = pics + 1
            logger.__load__(targets[i], prediction)
            label, score = logger.get_highest_predictions(score_limit=score_limit)
            data, targets_success, predict_success = logger.calc_accuracy(score, overlap_limit=0.5)
            Tpos, Fpos, Fneg, Tneg = data

            Tpos_vec+=Tpos
            Fpos_vec+=Fpos
            Fneg_vec+=Fneg
            Tneg_vec+=Tneg

            if targets_success is not None:
                for val in targets_success:
                    if val != 0:
                        success_vec.append(1)
                    else:
                        success_vec.append(0)

    success_percent = success_vec.count(1) / len(success_vec)
    
    # data = sum(accuracy_vec) / len(accuracy_vec)

    torch.set_num_threads(n_threads)

    # Set back in training mode
    model.train()

    return success_percent, Tpos_vec, Fpos_vec, Fneg_vec, Tneg_vec


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
