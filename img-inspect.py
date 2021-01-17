from components.evaluation.utils_binary_evaluation import evaluate_binary
from components.evaluation.utils_evaluator import LogHelpers
from components.data_loader.data_load_binary_eval import LoadImages
from components.data_loader.data_load import solar_panel_data
from components.neural_nets.NNClassifier import ChooseModel
import components.torchvision_utilities.transforms as T
import components.torchvision_utilities.utils as utils
from train import evaluate, load_configuration
import components.neural_nets.NNClassifier
import scipy.io as sci
import argparse as ap
import numpy as np
import torch
import glob
import copy
import cv2


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise ap.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise ap.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x


def binary(args, img_dir, mask_dir, device, model):
    data_loader = LoadImages(img_dir, mask_dir, normalize=True)
    print(f'Evaluating {len(data_loader)} images')

    data_loader_test = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    if args.cutoff_percent != None:
        cutoff_percent = restricted_float(args.cutoff_percent)
    else:
        cutoff_percent = 0.5

    print(f"Evaluating binary case with cut off percentage: {cutoff_percent}")

    success_percent, fault_correct, no_fault_correct, fault_images, no_fault_images, images = evaluate_binary(
        model, data_loader_test, device, 0.5)

    print(f'Succes percentage is: {success_percent}')
    print(f'Faults succesfully found: {fault_correct/fault_images}')
    print(
        f'No faults succesfully not found: {no_fault_correct/no_fault_images}')
    print(f'Total images: {images}')
    print(f'Images with faults: {fault_images}')
    print(f'Images without fault: {no_fault_images}')


def evaluate_labels(img_dir, mask_dir, device, model):
    dataset_test = solar_panel_data(
        img_dir,
        mask_dir,
        filter=True,
        mask="mask",
        csv=True,
        train=False,
        normalize=True
    )

    num_classes = dataset_test.get_number_of_classes()
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:])

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    accuracy, success_percent, Tpos, Fpos, Fneg, Tneg, data = evaluate(
        model, data_loader_test, device, show_plot=True, inv_transform=True)

    logger = LogHelpers()
    logger.print_data(data, success_percent, label='all')
    logger.print_data(data, success_percent, label='Crack A')
    logger.print_data(data, success_percent, label='Crack B')
    logger.print_data(data, success_percent, label='Crack C')
    logger.print_data(data, success_percent, label='Finger failure')


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--binary')
    parser.add_argument('--cutoff_percent')
    args = parser.parse_args()
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    folder_name = "solar_model_mask_fault-classification_20210114-093235"

    model_path = "./Results-folder/" + folder_name

    conf = load_configuration(model_path + "/model_conf.json")

    model = ChooseModel(conf["Model"], conf["Labels"], freeze=False)
    model.load_state_dict(torch.load(model_path + "/" + folder_name))
    model.to(device)

    # Initialize data loader
    img_dir = "./data/Serie1_CellsAndGT/CellsCorr/"
    mask_dir = "./data/Serie1_CellsAndGT/MaskGT/"

    if args.binary != None:
        binary(args, img_dir, mask_dir, device, model)
    else:
        evaluate_labels(img_dir, mask_dir, device, model)


if __name__ == "__main__":
    main()
