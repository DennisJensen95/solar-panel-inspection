import scipy.io as sci
import cv2
import glob
import numpy as np
import components.neural_nets.NNClassifier
from helpers import evaluate, load_configuration, get_transform
from components.neural_nets.NNClassifier import ChooseModel
import torch
import components.torchvision_utilities.utils as utils
import components.torchvision_utilities.transforms as T
from components.data_loader.data_load import solar_panel_data
import copy


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    folder_name = "solar_model_mask_binary_20210114-165706"

    model_path = "./Results-folder/" + folder_name

    conf = load_configuration(model_path + "/model_conf.json")

    model = ChooseModel(conf["Model"], conf["Labels"], freeze=False)
    model.load_state_dict(torch.load(model_path + "/" + folder_name))
    model.to(device)

    # ------ LOAD DATA ------------

    # Initialize data loader
    img_dir = "./data/Serie1_CellsAndGT/CellsCorr/"
    mask_dir = "./data/Serie1_CellsAndGT/MaskGT/"
    dataset_train = solar_panel_data(
        img_dir,
        mask_dir,
        filter=True,
        mask="mask",
        train=True,
        normalize=True,
        binary=False
    )
    dataset_test = copy.deepcopy(dataset_train)
    dataset_test.transforms = get_transform(train=False)

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

    accuracy, success_percent, Tpos, Fpos, Fneg, Tneg = evaluate(model, data_loader_test, device, show_plot=True, inv_norm=True)
    
    print(f'Targets found: {success_percent} percent')
    print(f'Mean accuracy: {accuracy}')
    print(f'Confusion matrix:')
    print(f'n={len(dataset_test)}        |    Predicted Yes   |   Predicted No')
    print(f'Actual Yes  |        {Tpos}          |       {Fneg}')
    print(f'Actual No   |        {Fpos}          |       {Tneg}')
    

if __name__ == "__main__":
    main()
