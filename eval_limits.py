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
import argparse as ap
import os
import pandas as pd


def setup_arg_parsing():
    parser = ap.ArgumentParser()
    parser.add_argument('--debug', help='debug flag help')
    parser.add_argument('--folder', help='Model folder path')
    parser.add_argument('--augment', help='Model folder path')
    args = parser.parse_args()
    
    return args

def write_data_csv(data, root_dir):
    filename = create_filename()

    data_frame = pd.DataFrame(data)
    data_frame.to_csv(root_dir + "/" + filename)

def create_filename():
    return "AUC_testdata"

def main():
    args = setup_arg_parsing()
    # debug = True if args.debug != None else False

    if args.folder == None:
        print(f'Please specify a model')
        return
    else:
        model_path = args.folder
        if not os.path.exists(model_path):
            print(f'Folder does not exist')
            return
        
        try:
            folder_name = os.path.basename(os.path.normpath(model_path)) # extract folder name
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            conf = load_configuration(model_path + "/model_conf.json")
            model = ChooseModel(conf["Model"], conf["Labels"], freeze=False)
            model.load_state_dict(torch.load(model_path + "/" + folder_name))
            model.to(device)
        except:
            print(f'Failed to load model')
            return


    # ------ LOAD DATA ------------

    # Initialize data loader
    if args.augment != None:
        img_dir = "./data/SerieA_CellsAndGT/CellsCorr/"
        mask_dir = "./data/SerieA_CellsAndGT/MaskGT/"
    else:
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
    dataset_test.train = False

    num_classes = dataset_test.get_number_of_classes()
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[:])

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    success_vec = []
    Tpos_vec = []
    Fpos_vec = []
    Fneg_vec = []
    Tneg_vec = []
    accuracy_vec = []

    lim1 = np.linspace(0.0,0.2,num=11)
    lim2 = np.linspace(0.25,0.8,num=12)
    lim3 = np.linspace(0.82,1.0,num=10)
    limits=np.concatenate((lim1,lim2,lim3))
    for limit in limits:
        print(f'Currently checking score limit: {limit}')
        success_percent, Tpos, Fpos, Fneg, Tneg = evaluate(
            model, 
            data_loader_test, 
            device, 
            show_plot=False, 
            inv_norm=True, 
            score_limit=limit
            )

        accuracy = (Tpos+Tneg)/(Tpos+Tneg+Fpos+Fneg)

        accuracy_vec.append(accuracy)
        success_vec.append(success_percent)
        Tpos_vec.append(Tpos)
        Fpos_vec.append(Fpos)
        Fneg_vec.append(Fneg)
        Tneg_vec.append(Tneg)
    
    data = {
                # "Num images": num_images,
                "Accuracy": accuracy_vec,
                "Succes Percentage": success_vec,
                "True positives": Tpos_vec,
                "False positives": Fpos_vec,
                "False negatives": Fneg_vec,
                "True negatives": Tneg_vec,
                "Limits": limits,
            }
    write_data_csv(data, model_path)
    

if __name__ == "__main__":
    main()
