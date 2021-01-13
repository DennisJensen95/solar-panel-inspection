import torchvision.transforms.functional as TF
from torchvision import transforms
from random import shuffle
import scipy.io as sci
from PIL import Image
import pandas as pd
import numpy as np
import random
import torch
import glob
import cv2
import csv
import os


class DataLoaderImageClassification:

    def __init__(self, img_dir, gt_dir, Filter=True, train=True):
        super().__init__()
        self.train = train
        self.ImageDir = img_dir + "*"
        self.GTDir = gt_dir + "*"

        self.data = self.load()

        self.csv_filepath = "./data/available_files.csv"

        if os.path.exists(self.csv_filepath):
            print("Load csv")
            self.load_csv()
        elif filter:
            print(f"Avaliable files: {len(self.data[0])}")
            # self.write_csv()def transform_torch_to_cv2(image, channels=3):

    def load(self):
        """Load path to png images and labels/masks"""
        return self.generate_path()

    def generate_path(self):
        """Generates path to images and removes
        any that does not have a corresponding mask

        Returns:
            [list]: [path to images]
            [list]: [path to masks/labels]
        """
        # Load paths using glob
        files = sorted(glob.glob(self.ImageDir))
        masks = sorted(glob.glob(self.GTDir))

        data = self.label_and_sort_images(files, masks)

        return data

    def write_csv(self):
        with open("./data/available_files.csv", "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CellsCorr", "Labels"])

            for i in range(len(self.data)):
                writer.writerow([self.data[0][i], self.data[1][i]])

    def load_csv(self):
        dataframe = pd.read_csv("./data/available_files.csv")
        self.files = dataframe["CellsCorr"]
        self.labels = dataframe["Labels"]

    def remove_no_matches(self, a, b, n_f, n_m):
        """Returns any path that does not have a corresponding mask

        Returns:
            [list]: [image paths with corresponding mask]
        """
        # Beware of the index!!
        names = [w[(n_f + 19): -4] for w in a]
        names_m = [w[(n_m + 18): -4] for w in b]

        # Only keep strings that occur in each list
        names = [x for x in names if x in names_m]

        # Add the path back and return
        return [a[0][: (n_f + 19)] + w + a[0][-4:] for w in names]

    def label_and_sort_images(self, files, masks):
        mask_with_fault = []
        mask_without_fault = []
        no_fault_images = files
        for i, mask in enumerate(masks):
            GT = sci.loadmat(mask)
            Labelstemp = GT["GTLabel"]  # fault labels

            # Make a list of all files that has a label and those without
            if Labelstemp.size > 0:
                mask_with_fault.append(mask)
            else:
                mask_without_fault.append(mask)
                

        n_f = len(self.ImageDir) - 1
        n_m = len(self.GTDir) - 1
        fault_images = self.remove_no_matches(files, mask_with_fault, n_f, n_m)
        no_fault_images = self.remove_no_matches(files, mask_without_fault, n_f, n_m)

        number_faults = len(fault_images)
        number_no_faults = len(no_fault_images)
        print(f'Number of images with faults: {number_faults}')
        print(f'Images with no fault {number_no_faults}')
        print(f'Reducing images with no fault size to equivalent of faults')

        data = self.make_data_structure(fault_images, files, number_faults, number_no_faults)

        return data

    def make_data_structure(self, images_with_error, no_fault_images, number_faults, number_no_faults):
        shuffle(no_fault_images)
        no_fault_images = no_fault_images[:number_faults +
                                                  int(0.3*number_faults)]

        label_error = list(np.ones((len(images_with_error))))
        label_no_error = list(np.zeros((len(no_fault_images))))

        data = [no_fault_images + images_with_error,
                label_no_error + label_error]

        return data

    def transform_data(self, image, train):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)

        return image

    def __getitem__(self, idx):
        img = self.data[0][idx]
        labels = self.data[1][idx]

        img = Image.open(img)
        img = img.convert('RGB')

        labels = torch.as_tensor([labels], dtype=torch.int64)

        img = self.transform_data(img, self.train)

        return img, labels

    def __len__(self):
        return len(self.data[0])


def transform_torch_to_cv2(image, channels=3):
    transform = transforms.ToPILImage()
    image = np.array(transform(image))
    image = np.reshape(image, (224, 224, channels))
    return image


if __name__ == "__main__":
    ImageDir = "data/Serie1_CellsAndGT/CellsCorr/"
    GTDir = "data/Serie1_CellsAndGT/MaskGT/"
    data_loader = DataLoaderImageClassification(ImageDir, GTDir, train=True)

    i = 0
    for image, label in iter(data_loader):
        image_cv2 = transform_torch_to_cv2(image)
        cv2.imshow("Target", image_cv2)
        cv2.waitKey(500)

        if i == 5:
            break
        i += 1
