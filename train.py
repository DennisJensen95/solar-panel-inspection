# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data, DisplayBoundingBoxes, DisplayMasks
from components.evaluation.utils_evaluator import LogHelpers
from components.neural_nets.NNClassifier import ChooseModel
import components.torchvision_utilities.utils as utils
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


results_folder = "Results-folder"


def create_filename(name, model, classification, timestamp=False):
    if not (timestamp != False):
        timestr = time.strftime("%Y%m%d-%H%M%S")
    else:
        timestr = timestamp
    return name + "_" + model + "_" + classification + "_" + timestr


def create_folder(name, configuration, conf_name="model_conf.json"):
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

    with open(path + "/" + conf_name, "w+") as file:
        json.dump(configuration, file)

    return path, timestr


def create_results_folder(configuration):
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


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

    success_array = []
    pics = 0

    data_iter_test = iter(data_loader_test)
    # iterate over test subjects
    for images, targets in data_iter_test:
        images = list(img.to(device) for img in images)

        # torch.cuda.synchronize()  # what is this??
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        for i, image in enumerate(outputs):
            pics = pics + 1
            logger.__load__(image, targets[i])
            label, score = logger.get_highest_predictions()
            success, targets_success, overlaps = logger.get_success_w_box_overlap(
                label, score
            )

            if success:
                for val in targets_success:
                    success_array.append(val)
            else:
                n_targ = len(targets[i]["labels"])
                for k in range(n_targ):
                    success_array.append(0)

    success_percent = success_array.count(1) / len(success_array)

    # Put back in train mode
    model.train()

    torch.set_num_threads(n_threads)

    return success_percent


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
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = ChooseModel(configuration["Model"],
                        configuration["Labels"], freeze=True)
    model.to(device)

    # Initialize data loader
    img_dir = "./data/Serie1_CellsAndGT/CellsCorr/"
    mask_dir = "./data/Serie1_CellsAndGT/MaskGT/"
    dataset_train = solar_panel_data(
        img_dir,
        mask_dir,
        filter=True,
        mask=configuration["Model"],
        train=True
    )

    dataset_test = copy.deepcopy(dataset_train)
    dataset_test.train = False

    num_classes = dataset_train.get_number_of_classes()
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=2,
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
    epochs = 15
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

    model.train()
    start = time.time()
    losses_data = []
    num_images = []
    time_data = []
    success_percentage_data = []
    success_percentage = 0
    start_time = time.time()
    for epoch in range(epochs):
        data_iter_train = iter(data_loader_train)
        for images, targets in data_iter_train:
            # Move images and targets to GPU
            images = list(image.to(device) for image in images)

            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Check how bad
            losses.backward()
            # Be better
            optimizer.step()

            # print statistics
            if i % 60 == 0:
                print(
                    f"Epoch: {epoch}: loss {losses} Accuracy: {success_percentage*100}"
                )

            if i % 30 == 0 and i != 0:
                success_percentage = evaluate(model, data_loader_test, device)

            success_percentage_data.append(success_percentage)

            # Save data
            if torch.cuda.is_available():
                losses_data.append(losses.cpu().detach().numpy())
            else:
                losses_data.append(losses.detach().numpy())

            if len(num_images) > 1:
                num_images.append(num_images[-1] + len(images))
            else:
                num_images.append(len(images))
            time_data.append(time.time() - start_time)

            i += 1

        # Optimize learning rate
        lr_scheduler.step()

    filename = create_filename(
        "solar_model",
        configuration["Model"],
        configuration["Classification"],
        timestamp=time_stamp,
    )
    torch.save(model.state_dict(), root_dir + "/" + filename)

    data = {
        "Time": time_data,
        "Num images": num_images,
        "Loss": losses_data,
        "Succes Percentage": success_percentage_data,
    }
    write_data_csv(configuration, data, root_dir, timestamp=time_stamp)


if __name__ == "__main__":
    train()
