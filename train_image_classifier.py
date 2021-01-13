from components.data_loader.image_classification_data_loader import DataLoaderImageClassification
from components.neural_nets.ImageClassifier import ChooseModel
import components.torchvision_utilities.utils as utils
from train import setup_arg_parsing, create_folder, create_filename, load_configuration, write_data_csv
import numpy as np
import torch
import time
import copy

@torch.no_grad()
def evaluate(model, data_loader_test, device):
    success_array = []
    pics = 0

    data_iter_test = iter(data_loader_test)
    # iterate over test subjects
    total_success = 0
    total_evaluated = 0
    for images, labels in data_iter_test:
        predictions = []
        for j, image in enumerate(images):
            image = image.unsqueeze(0).to(device)
            label = labels[j].to(device)
            output = model(image)
            prediction = output.to("cpu").max(1)[1]
            predictions.append(prediction.numpy()[0])
        
        success = np.array(labels) == np.array(predictions[0])
        success_count = np.sum(success)
        
        total_success += success_count
        total_evaluated += len(labels)
        
    success_percent = total_success / total_evaluated

    return success_percent

def train():
    args = setup_arg_parsing()
    debug = True if args.debug != None else False
    configuration = load_configuration("simple_model_conf.json")
    root_dir, time_stamp = create_folder("solar_model", configuration, conf_name="simple_model_conf.json")
    # Locate cpu or GPU
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = ChooseModel(n_classes=configuration["Labels"], pretrained=True)
    model.to(device)

    # Initialize data loader
    img_dir = "./data/combined_data/CellsCorr/"
    mask_dir = "./data/combined_data/MaskGT/"
    dataset_train = DataLoaderImageClassification(
        img_dir,
        mask_dir,
        train=True,
        Filter=True
    )

    dataset_test = copy.deepcopy(dataset_train)
    dataset_test.train = False

    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-100])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-100:])

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=50,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # Predefined values
    epochs = 30
    i = 0

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(
        params,
        lr=configuration["LearningRate"],
        weight_decay=configuration["WeightDecay"],
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    model.train()
    losses_data = []
    num_images = []
    time_data = []
    success_percentage_data = []
    success_percentage = 0
    start_time = time.time()
    for epoch in range(epochs):
        data_iter_train = iter(data_loader_train)
        i = 0
        running_loss = 0
        for images, labels in data_iter_train:
            # Move images and targets to GPU
            optimizer.zero_grad()
            for j, image in enumerate(images):
                image = image.unsqueeze(0).to(device)
                label = labels[j].to(device)
                output = model(image)
                loss = criterion(output, label)
                loss.backward()

            optimizer.step()
            running_loss += loss.item()
            
            if torch.cuda.is_available():
                losses_data.append(running_loss)
            else:
                losses_data.append(running_loss)

            if len(num_images) > 1:
                num_images.append(num_images[-1] + len(images))
            else:
                num_images.append(len(images))
            time_data.append(time.time() - start_time)
            i += 1
            
        success_percentage = evaluate(model, data_loader_test, device)
        print(
            f"Epoch: {epoch}: loss {running_loss/i} Accuracy: {success_percentage*100}"
        )

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
