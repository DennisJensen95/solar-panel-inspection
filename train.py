# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data
import components.torchvision_utilities.transforms as T
import components.torchvision_utilities.utils as utils
import torchvision
import torch
import copy
import time
import os

results_folder = "Results-folder"


def create_filename_timestamp(name):
    timestr = time.strftime("%Y%m%d-%H%M")
    return name + "_" + timestr


def create_results_folder():
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train():
    create_results_folder()
    # Locate cpu or GPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)

    # Initialize data loader
    img_dir = "./data/Serie1_CellsAndGT/CellsCorr/"
    mask_dir = "./data/Serie1_CellsAndGT/MaskGT/"
    dataset_train = solar_panel_data(
        img_dir, mask_dir, filter=True, transforms=get_transform(train=True)
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

    # Predefined values
    epochs = 1
    i = 0

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.train()
    start = time.time()
    data_iter_train = iter(data_loader_train)
    data_iter_test = iter(data_loader_test)
    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in data_iter_train:
            # Move images and targets to GPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Check how bad
            losses.backward()

            # Be better
            optimizer.step()

            # print statistics
            print(i)
            if i % 60 == 0:
                print(f"Epoch: {epoch}: loss {losses}")

            i += 1

    filename = create_filename_timestamp("solar_model")
    torch.save(model.state_dict(), "./" + results_folder + "/" + filename)


if __name__ == "__main__":
    train()