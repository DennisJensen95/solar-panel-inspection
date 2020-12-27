import torchvision.models as models
from torchvision import transforms
import torch
import sys
from PIL import Image

alexnet = models.alexnet(pretrained=True)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]
)


def setup_data(img_path):
    img = Image.open(img_path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    return batch_t


def get_existing_labels():
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    return classes


def get_highest_prediction(network_predictions):
    labels = get_existing_labels()
    _, index = torch.max(network_predictions, 1)
    percentage = torch.nn.functional.softmax(
        network_predictions, dim=1)[0] * 100
    label = labels[index[0]]
    percentage_certainty = percentage[index[0]].item()
    print(percentage_certainty)

    return (label, percentage_certainty)


def predict_dogs(img_path):
    data = setup_data(img_path)
    alexnet.eval()
    out = alexnet(data)
    label, percentage_certainty = get_highest_prediction(out)
    print(f'Dog race is: {label}')
    print(f'Certainty is: {percentage_certainty}')


if __name__ == "__main__":
    filename = sys.argv[1]
    predict_dogs(filename)
