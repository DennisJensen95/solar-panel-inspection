from ..visualize.visualize_faults import visualize_faults
from ..data_load.data_load import transform_image
from ..model_load.module_load import ChooseModel
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import torch


def load_model(output_neurons=5):
    model = ChooseModel("mask", output_neurons)
    return model


def setup_model(pretrained_model_filename):
    model = load_model()
    model.load_state_dict(torch.load(pretrained_model_filename))
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(device)
    # model.to(device)
    model.eval()

    return model, device


def predict_faults(image, pretrained_model_filename):
    model, device = setup_model(pretrained_model_filename)
    image_predict = transform_image(image, device)
    outputs = model(image_predict)
    outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
    visualize_faults(image_predict[0], outputs[0])
