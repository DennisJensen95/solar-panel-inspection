# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data, DisplayBoundingBoxes, DisplayMasks
import components.torchvision_utilities.utils as utils
from components.evaluation.utils_evaluator import LogHelpers
from components.neural_nets.NNClassifier import ChooseModel
import argparse as ap
import torchvision
import pandas as pd
import numpy as np
import torch
import json
import copy
import time
import cv2
import os
import math
import sys


results_folder = "Results-folder"


def create_filename(name, model, classification, timestamp=False):
    if not (timestamp != False):
        timestr = time.strftime("%Y%m%d-%H%M%S")
    else:
        timestr = timestamp
    return name + "_" + model + "_" + classification + "_" + timestr


def create_folder(name, configuration):
    create_results_folder(configuration)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    folder_name = (
        name
        + "_"
        + configuration["Model"]
        + "_"
        + configuration["Classification"]
        + "_"
        + timestr
    )
    path = results_folder + "/" + folder_name
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/model_conf.json", "w+") as file:
        json.dump(configuration, file)

    return path, timestr


def create_results_folder(configuration):
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

def train_one_epoch(model, optimizer, data_loader, data_loader_test, device, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    i = 0
    losses_val = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        losses_val += losses
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        i += 1

    return losses_val/i

@torch.no_grad()
def evaluate(model, data_loader_test, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # Metric logger class
    logger = LogHelpers()

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
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for i, prediction in enumerate(outputs):
            pics = pics + 1
            logger.__load__(targets[i], prediction)
            label, score = logger.get_highest_predictions(score_limit=0.5)
            data, targets_success, predict_success = logger.calc_accuracy(score, overlap_limit=0.5)
            Tpos, Fpos, Fneg, Tneg = data
            
            try:
                accuracy = (Tpos+Tneg)/(Tpos+Tneg+Fpos+Fneg)
            except RuntimeWarning:
                print('Cannot compute accuracy')
                print(data)
                accuracy = 0.0
            
            accuracy_vec.append(accuracy)
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

    return sum(accuracy_vec)/len(accuracy_vec), success_percent, Tpos_vec, Fpos_vec, Fneg_vec, Tneg_vec


def load_configuration(filename=None):
    if filename == None:
        filename = "model_conf.json"

    with open(filename, "r") as file:
        configuration = json.load(file)

    print(f'Selected model: {configuration["Model"]}')
    print(f'Label configuration: {configuration["Classification"]}')

    return configuration


def write_data_csv(configuration, data, root_dir, timestamp=False):
    filename = create_filename(
        "solar_model_data",
        configuration["Model"],
        configuration["Classification"],
        timestamp
    )

    data_frame = pd.DataFrame(data)
    data_frame.to_csv(root_dir + "/" + filename)

def setup_arg_parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('--debug', help='debug flag help')
    args = parser.parse_args()
    
    return args

def train():
    args = setup_arg_parsing()
    debug = True if args.debug != None else False
    configuration = load_configuration()
    root_dir, time_stamp = create_folder("solar_model", configuration)
    # Locate cpu or GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ChooseModel(configuration["Model"], configuration["Labels"], freeze=False)
    model.to(device)

    # Initialize data loader
    img_dir = "./data/SerieA_CellsAndGT/CellsCorr/"
    mask_dir = "./data/SerieA_CellsAndGT/MaskGT/"
    dataset_train = solar_panel_data(
        img_dir,
        mask_dir,
        filter=True,
        mask=configuration["Model"],
        train=True,
        normalize=False
    )

    dataset_test = copy.deepcopy(dataset_train)
    dataset_test.train = False

    num_classes = dataset_train.get_number_of_classes()
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=configuration["BatchSize"],
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    
    if debug and configuration["Model"] == "Faster":
        for images, targets in iter(data_loader_train):
            DisplayBoundingBoxes(images[0], targets[0]["boxes"], 5)
    elif debug and configuration["Model"] == "mask":
        for images, targets in iter(data_loader_train):
            DisplayBoundingBoxes(images[0], targets[0]["boxes"], 5)
            # print(len(targets))
            DisplayMasks(images[0], targets[0])
            
    
    # Predefined values
    epochs = 5
    i = 0

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(
        params,
        lr=configuration["LearningRate"],
        weight_decay=configuration["WeightDecay"],
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=configuration["StepSize"], gamma=configuration["Gamma"]
    )

    start = time.time()
    losses_data = []
    num_images = []
    time_data = []
    accuracy_vec = []
    success_percent_vec = []
    Tpos_vec = []
    Fpos_vec = []
    Fneg_vec = []
    Tneg_vec = []

    start_time = time.time()
    for epoch in range(epochs):
        data_iter_train = iter(data_loader_train)

        losses = train_one_epoch(model, optimizer, data_iter_train, data_loader_test, device, epoch)
    
        # Optimize learning rate
        lr_scheduler.step()

        accuracy, success_percent, Tpos, Fpos, Fneg, Tneg = evaluate(model, data_loader_test, device)
    
        print(f"Epoch: {epoch}: loss {losses}")

        print(f'Targets found: {success_percent*100} percent')
        print(f'Mean accuracy: {accuracy*100} percent')
        print(f'Confusion matrix:')
        print(f'n={len(dataset_test)}        |    Predicted Yes   |   Predicted No')
        print(f'Actual Yes  |        {Tpos}          |       {Fneg}')
        print(f'Actual No   |        {Fpos}          |       {Tneg}')

        accuracy_vec.append(accuracy)
        success_percent_vec.append(success_percent)
        Tpos_vec.append(Tpos)
        Fpos_vec.append(Fpos)
        Fneg_vec.append(Fneg)
        Tneg_vec.append(Tneg)

        # Save data
        if torch.cuda.is_available():
            losses_data.append(losses.cpu().detach().numpy())
        else:
            losses_data.append(losses.detach().numpy())
        
        time_data.append(time.time() - start_time)

    filename = create_filename(
        "solar_model",
        configuration["Model"],
        configuration["Classification"],
        timestamp=time_stamp,
    )
    torch.save(model.state_dict(), root_dir + "/" + filename)

    data = {
        "Time": time_data,
        # "Num images": num_images,
        "Loss": losses_data,
        "Accuracy": accuracy_vec,
        "Succes Percentage": success_percent_vec,
        "True positives": Tpos_vec,
        "False positives": Fpos_vec,
        "False negatives": Fneg_vec,
        "True negatives": Tneg_vec,
    }
    write_data_csv(configuration, data, root_dir, timestamp=time_stamp)


if __name__ == "__main__":
    train()
