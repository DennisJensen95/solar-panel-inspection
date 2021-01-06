import os
from torchvision import models
from  import data_load
import torch


model = model.vgg16(pretrained=True)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
)

os.mkdir("Results-Folder")
with open("./Results-Folder/result.txt", "wb") as file:
    file.write("Stefan er nice og det samme er Martin")
