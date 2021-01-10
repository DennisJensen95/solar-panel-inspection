# from components.torchvision_utilities.engine import train_one_epoch, evaluate
from components.data_loader.data_load import solar_panel_data
import components.torchvision_utilities.transforms as T
import components.torchvision_utilities.utils as utils
from components.evaluation.utils_evaluator import LogHelpers
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

    print(f"Entered evaluation")

    success_array = []
    pics = 0

    data_iter_test = iter(data_loader_test)
    # iterate over test subjects
    for images, targets in data_iter_test:
        images = list(img.to(device) for img in images)

        # torch.cuda.synchronize()  # what is this??
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        for i, image in enumerate(outputs):
            pics = pics + 1
            logger.__load__(image, targets[i])
            label, score = logger.get_highest_predictions()
            success, targets_success, overlaps = logger.get_success_w_box_overlap(label, score)

            if success:
                # print(f'Targets_success: {targets_success}')
                # print(f'Overlaps: {overlaps}')
                for val in targets_success:
                    success_array.append(val)
            else:
                n_targ = len(targets[i]['labels'])
                for k in range(n_targ):
                    success_array.append(0)
    
    print(f'Success (length={len(success_array)}): {success_array}')
    success_percent = success_array.count(1)/len(success_array)
    print(f'Percent success: {success_percent}')

    print(f'Pictures: {pics}')



            # print(f'Highest prediction label(s): {label}')
            # print(f'Score(s): {score}')

            


            # check overlapping area for bounding boxes
            

        # res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # evaluator_time = time.time()
        # # coco_evaluator.update(res)
        # evaluator_time = time.time() - evaluator_time
        # # metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Put back in train mode
    model.train()

    torch.set_num_threads(n_threads)

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

            if i % 5 == 0 and i != 0:
                evaluate(model, data_loader_test, device)

            i += 1

    filename = create_filename_timestamp("solar_model")
    torch.save(model.state_dict(), "./" + results_folder + "/" + filename)


if __name__ == "__main__":
    train()