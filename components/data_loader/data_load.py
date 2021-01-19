#!/usr/bin/python3

import scipy.io as sci
import cv2
import glob
import numpy as np
import os
import csv
import pandas as pd
import torch
import os
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
from PIL import Image


class solar_panel_data:
    def __init__(
        self,
        img_dir,
        gt_dir,
        train,
        filter=True,
        csv=True,
        area_limit=100,
        mask="faster",
        normalize=False,
        binary=False
    ):
        if mask == "mask":
            self.mask = True
        else:
            self.mask = False

        self.ImageDir = img_dir + "*"
        self.GTDir = gt_dir + "*"

        self.files, self.masks = self.Load()
        

        self.label_dic = LabelEncoder(binary=binary)
        self.normalize = normalize
        
        self.csv_filepath = "./data/available_files.csv"
        
        self.train = train
        
        self.norm_mean = (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)
        
        if os.path.exists(self.csv_filepath) and csv:
            print("Load csv")
            self.load_csv()
        
        elif filter:
            print("Removing files without labels from path...")
            print(f"Avaliable files: {len(self.files)}")
            self.RemoveNoLabels()
            print(f"Avaliable files: {len(self.files)}")
            print(
                f"Removing files where all areas are smaller than {area_limit} square pixel..."
            )
            self.RemoveErrors(area_limit)
            print(f"Avaliable files: {len(self.files)}")
            self.print_csv()    
        

    def Load(self):
        """Load path to png images and labels/masks"""
        return self.GeneratePath()

    def GeneratePath(self):
        """Generates path to images and removes
        any that does not have a corresponding mask

        Returns:
            [list]: [path to images]
            [list]: [path to masks/labels]
        """

        n_f = len(self.ImageDir) - 1
        n_m = len(self.GTDir) - 1

        # Load paths using glob
        files = sorted(glob.glob(self.ImageDir))
        masks = sorted(glob.glob(self.GTDir))

        files = self.RemoveNoMatches(files, masks, n_f, n_m)

        return files, masks

    def CleanErrors(self, filename, mask):
        """Special case where data can be fixed by simply removing some numbers

        Args:
            filename (string): the image path of the file in question
            mask (numpy array): the given mask

        Returns:
            [numpy array]: fixed mask
        """
        # print(filename[-27:])
        if filename[-27:] == "10_4081_Cell_Row5_Col_3.png":
            mask[114:119, 264:277] = 0
            # print("Found bad mask. Doing cleanup...")

        return mask

    def RemoveNoMatches(self, a, b, n_f, n_m):
        """Returns any path that does not have a corresponding mask

        Returns:
            [list]: [image paths with corresponding mask]
        """
        # Beware of the index!!
        names = [w[(n_f + 19) : -4] for w in a]
        names_m = [w[(n_m + 18) : -4] for w in b]

        # Only keep strings that occur in each list
        names = [x for x in names if x in names_m]

        # Add the path back and return
        return [a[0][: (n_f + 19)] + w + a[0][-4:] for w in names]

    def RemoveNoLabels(self):
        """Removes measurements that does not contain a label"""

        names_m = []
        flag = False
        for i in range(len(self.masks)):
            GT = sci.loadmat(self.masks[i])
            Labelstemp = GT["GTLabel"]  # fault labels

            # if self.masks[i][-27:] == "10_4081_Cell_Row5_Col_4.mat":
            #     print("Remove bad measurement")
            #     flag = True

            # Make a list of all files that has a label
            if Labelstemp.size > 0 and not flag:
                names_m.append(self.masks[i])

            flag = False

        # Update mask list
        self.masks = names_m
        n_f = len(self.ImageDir) - 1
        n_m = len(self.GTDir) - 1

        # Update files list
        self.files = self.RemoveNoMatches(self.files, names_m, n_f, n_m)

    def RemoveErrors(self, area_limit):
        """Remove measurements where the area in a mask is too small.
        The function also checks for errors in the given data such as
        disparity between number of labels and amount of unique masks

        Args:
            area_limit (integer): limit for removing bad areas
        """
        names_m = []
        for i in range(len(self.files)):
            if i == len(self.files) - 1:
                print(".")
            if i % 50 == 0:
                print(".", end="", flush=True)

            if not self.__getitem__(i, find_error=True, area_limit=area_limit):
                names_m.append(self.masks[i])

        self.masks = names_m
        n_f = len(self.ImageDir) - 1
        n_m = len(self.GTDir) - 1
        self.files = self.RemoveNoMatches(self.files, names_m, n_f, n_m)

    def PrintPaths(self):
        for i in range(len(self.files)):
            print(self.files[i])
            print(self.masks[i])
            print("------------------------")

    def ResizeMasks(self, masks, resized_size):
        resized_mask = np.resize(masks, (len(masks), resized_size[0], resized_size[1]))
        return resized_mask

    def ResizeBoundingBox(self, boxes, orig_shape, resized_size):
        resized_boxes = []
        for box in boxes:
            target_shape = orig_shape

            x_scale = resized_size[0] / target_shape[0]
            y_scale = resized_size[1] / target_shape[1]

            xmin = int(np.round(box[0] * x_scale))
            ymin = int(np.round(box[1] * y_scale))
            xmax = int(np.round(box[2] * x_scale))
            ymax = int(np.round(box[3] * y_scale))

            resized_boxes.append([xmin, ymin, xmax, ymax])

        return resized_boxes
    

    def transform_mask(self, image, masks, train):
        # Resize
        # print(f'Before resize: {np.shape(np.array(masks))}')

        resize = transforms.Resize(size=(224, 224))
        norm = transforms.Normalize(self.norm_mean, self.norm_std)
        image = TF.to_tensor(image)
        image = resize(image)
        if self.normalize:
            image = norm(image)
        
        new_masks = []
        for i, mask in enumerate(masks):
            new_masks.append(resize(mask))
        masks = new_masks

        # Transform to tensor
        new_masks = []
        for i, mask in enumerate(masks):
            # new_masks.append(TF.to_tensor(mask))
            new_masks.append(np.asarray(mask))
        
        # masks = TF.to_tensor(np.array(new_masks))

        masks = torch.as_tensor(np.asarray(new_masks), dtype=torch.uint8)
        # masks = new_masks
        
        return image, masks
    
    def transform_image(self, image, train):
        # Resize
        resize = transforms.Resize(size=(224, 224))
        image = resize(image)
        
        # Random horizontal flipping
        if random.random() > 0.5 and train:
            image = TF.hflip(image)

        # Random vertical flipping
        if random.random() > 0.5 and train:
            image = TF.vflip(image)
            
        # Transform to tensor
        image = TF.to_tensor(image)
        
        return image

    def __getitem__(self, idx, find_error=False, area_limit=100):
        """Method to load data
        Function returns independent variable (image) and dependent variable (target).

        The dependent variable contains various information:
                target["boxes"]
                target["labels"]
                target["label_str"]
                target["masks"]
                target["image_id"]
                target["area"]

        The function also tests and removes masks that have an area smaller than a limit value

        The "find_error" flag, when set to True will NOT return the variables, but instead return
        a True or False value, whether or not the data is usable or not. This is used in the
        RemoveErrors() function.

        Args:
            idx ([type]): Index of the file we are inspecting
            find_error (bool, optional): Flag to check for errors. Defaults to False.
            area_limit (int, optional): Defaults to 100.
        """
        # load images and masks
        img_path = self.files[idx]
        mask_path = self.masks[idx]

        # Load .mat file
        GT = sci.loadmat(mask_path)
        Labelstemp = GT["GTLabel"]  # fault labels
        Labels = np.transpose(Labelstemp)
        mask = GT["GTMaskOld"]  # fault mask
        # Load example image

        img = Image.open(img_path)
        img = img.convert('RGB')
        orig_img_size = np.shape(img)

        # Check if this mask is found to be errenous
        mask = self.CleanErrors(img_path, mask)

        # If there are no labels, we assume no faults in the image
        if Labels.size == 0:
            new_labels = []
            labels = []
            boxes = [[]]
            masks = mask

        else:  # Start checking for bounding boxes
            # Construct an array of labels
            new_labels = []
            for i in range(len(Labels[0])):
                new_labels.append(Labels[0][i][0])

            # 0: background
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]
            masks = masks.astype(np.uint8)
            

            # Checks for disparity between number of unique numbers in the mask and amount of labels
            if find_error and (len(new_labels) is not len(obj_ids)):
                if (
                    (len(new_labels) > len(obj_ids))
                    and len(obj_ids) != 0
                    and len(new_labels) != 0
                ):
                    newnewlabels = []
                    for obj in obj_ids:
                        newnewlabels.append(new_labels[obj - 1])

                    obj_ids = np.array(list(range(1, 1 + len(newnewlabels))))
                    new_labels = newnewlabels
                else:
                    return True

            # get bounding box coordinates for each mask
            boxes = []
            for i in range(len(obj_ids)):
                cnt = cv2.findContours(
                    masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[-2]
                for c in cnt:
                    x, y, w, h = cv2.boundingRect(c)
                    xmin = x
                    xmax = x + w
                    ymin = y
                    ymax = y + h
                    boxes.append([xmin, ymin, xmax, ymax])

        boxes = self.ResizeBoundingBox(boxes, orig_img_size, (224, 224))

        # Checks for disparity between number of bounding boxes and unique numbers
        # I.e. if two clusters in the mask has the same number
        if find_error and (len(boxes) is not len(obj_ids)):
            return True

        # Convert to tensor to calculate area easily
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Area check (if there are any labels/areas)
        if Labels.size > 0:
            newboxes = []
            newlabels = []
            newmasks = []
            for i in range(len(masks)):
                if area[i].numpy() > area_limit:
                    newboxes.append(boxes[i].numpy())
                    newlabels.append(new_labels[i])
                    newmasks.append(masks[i])

            boxes = newboxes
            new_labels = newlabels
            masks = newmasks

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(self.__getlabel__(new_labels), dtype=torch.int64)
        image_id = torch.tensor([idx])

        # Calculate new area matrix
        if find_error:
            if len(new_labels) > 0:
                if find_error:
                    return False
                area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            else:  # If no more areas are left that can be used, we must let the program know to completely remove the file
                boxes = torch.as_tensor([[]], dtype=torch.float32)
                area = []

                if find_error:
                    return True

        new_masks = []
        for i, mask in enumerate(masks):
            new_masks.append(Image.fromarray(mask))
        
        masks = new_masks
        
        if self.mask:
            img, masks = self.transform_mask(img, masks, self.train)
        else:
            img = self.transform_image(img, self.train)
            
        # Construct target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.mask:
            target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area  

        return img, target

    def __len__(self):
        return len(self.files)

    def __getlabel__(self, labels):
        """Extract numerical label from label string

        Args:
            labels (numpy array): array of array of strings

        Returns:
            numpy array: array of numerical labels
        """

        lab = []
        for i in range(len(labels)):
            #lab.append(self.label_dic[labels[i]])
            lab.append(self.label_dic.encode(labels[i]))
            #print(lab)

        return lab

    def get_number_of_classes(self):
        #print(f"length is: {len(self.label_dics.fault_key_to_value)}")
        return len(self.label_dic.fault_key_to_value)

    def print_csv(self):
        with open("./data/available_files.csv", "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CellsCorr", "MaskGT"])

            for i in range(len(self.files)):
                writer.writerow([self.files[i], self.masks[i]])
    
    def load_csv(self):
        dataframe = pd.read_csv("./data/available_files.csv")
        self.files = dataframe["CellsCorr"]
        self.masks = dataframe["MaskGT"]

def inv_normalize(img):
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ], 
                                                         std = [ 1/norm_std[0], 1/norm_std[1], 1/norm_std[2] ]),
                                    transforms.Normalize(mean = [ -norm_mean[0], -norm_mean[1], -norm_mean[2] ], 
                                                         std = [ 1., 1., 1. ]),
                                    ])
    img = invTrans(img)
    
    return img

class LabelEncoder:

    def __init__(self, binary=False):
        if binary:
            self.fault_value_to_key = {'Crack A': 1,
                                       'Crack B': 1,
                                       'Crack C': 1,
                                       'Finger Failure' : 1}
        
        else:
            self.fault_value_to_key = {'Crack A': 1,
                                       'Crack B': 2,
                                       'Crack C': 3,
                                       'Finger Failure' : 4}
                                       # 'No failure' : 2}

    

        self.fault_key_to_value = {v: k for k, v in self.fault_value_to_key.items()}


    def encode(self,input):
        return self.fault_value_to_key[input]

    def decode(self,input):
        return self.fault_key_to_value[input]


def DisplayTargetMask(mask, idx):
    mask = transform_torch_to_cv2(mask[idx], 1)
    mask = mask * 255

    # Show image
    cv2.imshow("Image", mask)
    # Show image
    # cv2.imshow('Image', im)
    cv2.waitKey(500)
    cv2.destroyAllWindows()


def DisplayImg(im):
    # Show image
    cv2.imshow("Image", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def DisplayAllFaults(masks):
    for i in range(len(masks)):
        GT = sci.loadmat(masks[i])
        Labelstemp = GT["GTLabel"]  # fault labels

        if Labelstemp.size > 0:
            print(Labelstemp)
            print(i)

def DisplayBoundingBoxes(im, boxes, time):
    boxes = boxes.numpy().astype(np.uint32)
    im = transform_torch_to_cv2(im)
    if len(boxes[0]) > 0:
        for i in range(len(boxes)):
            cv2.rectangle(
                im,
                (boxes[i][0], boxes[i][1]),
                (boxes[i][2], boxes[i][3]),
                (0, 255, 0),
                2,
            )

            xc = boxes[i][2] / 2 + boxes[i][0] / 2
            if np.abs(xc) > np.abs(xc - im.shape[0]):
                xc = (xc - 50).astype(np.uint64)
            else:
                xc = (xc + 15).astype(np.uint64)
            yc = ((boxes[i][3] / 2 + boxes[i][1] / 2)).astype(np.uint64)
            # print(f"(xc,yc) = ({xc},{yc})")
    
    cv2.imshow("Image boxes", im)
    cv2.waitKey(500)

def DisplayMasks(im, target):
    masks = target["masks"]
    for i in range(len(masks)):
        DisplayTargetMask(masks, i)

def transform_torch_to_cv2(image, channels=3):
    transform = transforms.ToPILImage()
        
    image = np.array(transform(image))
    # print(np.shape(image))
    image = np.reshape(image, (224, 224, channels))
    
    return image
    
def main():
    # os.chdir('components')
    # Path to directories
    ImageDir = "data/combined_data/CellsCorr/"
    GTDir = "data/combined_data/MaskGT/"

    data_serie1 = solar_panel_data(ImageDir, GTDir, filter=True, mask="mask", train=False)
    
    # num = 8499#8500#8512#8511#8697 #4494  
    num = 11
    im, target = data_serie1.__getitem__(num)
    im = transform_torch_to_cv2(im)
    #data_serie1.get_number_of_classes()
    print(f'Fault label: {target["labels"].numpy()}')
    boxes = target["boxes"].numpy().astype(np.uint32)
    print(f"Boxes: {boxes}")
    print(f'Area: {target["area"]}')

    if len(boxes[0]) > 0:
        for i in range(len(boxes)):
            print(im.shape)
            cv2.rectangle(
                im,
                (boxes[i][0], boxes[i][1]),
                (boxes[i][2], boxes[i][3]),
                (0, 255, 0),
                2,
            )

            xc = boxes[i][2] / 2 + boxes[i][0] / 2
            if np.abs(xc) > np.abs(xc - im.shape[0]):
                xc = (xc - 50).astype(np.uint64)
            else:
                xc = (xc + 15).astype(np.uint64)
            yc = ((boxes[i][3] / 2 + boxes[i][1] / 2)).astype(np.uint64)
            print(f"(xc,yc) = ({xc},{yc})")

            # DisplayTargetMask(target, i)

    # DisplayAllFaults(data_serie1.masks)

    DisplayImg(im)


if __name__ == "__main__":
    main()
