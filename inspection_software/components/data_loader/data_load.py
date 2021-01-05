#!/usr/bin/python3

import scipy.io as sci
import cv2
import glob
import numpy as np
import pandas as pd
from fastai.vision.all import *

class solar_panel_data():

    def __init__(self, img_dir, gt_dir):
        self.ImageDir = img_dir + "*"
        self.GTDir = gt_dir + "*"

        self.files, self.masks = self.Load()


    def Load(self):
        """Load path to png images and labels/masks
        """
        return self.GeneratePath()


    def GeneratePath(self):
        """Generates path to images and removes 
        any that does not have a corresponding mask 

        Returns:
            [list]: [path to images]
            [list]: [path to masks/labels]
        """

        n_f = len(self.ImageDir)-1
        n_m = len(self.GTDir)-1

        # Load paths using glob
        files = sorted(glob.glob(self.ImageDir))
        masks = sorted(glob.glob(self.GTDir))

        files = self.RemoveNoMatches(files,masks, n_f, n_m)

        return files, masks


    def RemoveNoMatches(self, a,b, n_f, n_m):
        """Returns any path that does not have a corresponding mask 

        Returns:
            [list]: [image paths with corresponding mask]
        """

        # Beware of the index!!
        names = [w[(n_f+19):-4] for w in a]
        names_m = [w[(n_m+18):-4] for w in b]

        # Only keep strings that occur in each list
        names = [x for x in names if x in names_m]

        # Add the path back and return
        return [a[0][:(n_f+19)] + w + a[0][-4:] for w in names]

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.files[idx]
        mask_path = self.masks[idx]

        # Load .mat file
        GT = sci.loadmat(mask_path)
        Labelstemp = GT['GTLabel'] # fault labels
        Labels = np.transpose(Labelstemp)
        mask = GT['GTMask'] # fault mask
        # Load example image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        new_labels = ["No Failure"]
        for i in range(len(Labels[0])):
            new_labels.append(Labels[0][i][0])
        
        # 0: background
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        masks = masks.astype(np.uint8)
        
        # get bounding box coordinates for each mask
        boxes = []
        for i in range(len(obj_ids)):
            cnt = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for c in cnt:
                x,y,w,h = cv2.boundingRect(c)
                xmin = x
                xmax = x+w
                ymin = y
                ymax = y+h
                boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.as_tensor(self.__getlabel__(new_labels), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["label_str"] = new_labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img, target


    def __getlabel__(self, labels):
        """Extract numerical label from label string

        Args:
            labels (numpy array): array of array of strings

        Returns:
            numpy array: array of numerical labels
        """    
        label_dic = {
            'Crack A': 1,
            'Crack B': 2,
            'Crack C': 3,
            'Finger Failure': 4,
            'No Failure': 0
        }

        lab = []
        for i in range(len(labels)):
            lab.append(label_dic[labels[i]])

        return lab



# def get_bounding_box(x) :
#     result = [[100.0,100.0,200.0,200.0]] # dummy values obviously
#     print("get_bounding_box(x)", x, "result", result)
#     return result

# def get_label(x):
#     result = ["bla"] # dummy values obviously
#     print("get_label(x)", x, "result", result)
#     return result


def DisplayTargetMask(target, idx):
    mask = target["masks"][idx].numpy()
    mask = mask * 255
    # Show image
    cv2.imshow('Image', mask)
    # Show image
    # cv2.imshow('Image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def DisplayImg(im):
    # Show image
    cv2.imshow('Image', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def DisplayAllFaults(masks):
    for i in range(len(masks)):
        GT = sci.loadmat(masks[i])
        Labelstemp = GT['GTLabel'] # fault labels

        if Labelstemp.size > 0:
            print(Labelstemp)
            print(i)

def main():
    # Path to directories
    ImageDir = "../data/Serie1_CellsAndGT/CellsCorr/"
    GTDir = "../data/Serie1_CellsAndGT/MaskGT/"

    dl = solar_panel_data(ImageDir, GTDir)

    num = 4494#8697 #4494
    im, target = dl.__getitem__(num)

    print(target["label_str"])
    print(target["labels"])

    boxes = target["boxes"].numpy().astype(np.uint32)
    for i in range(len(boxes)):
        cv2.rectangle(im,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,255,0),2)
        if i == 0:
            cv2.putText(im,target["label_str"][i], (100,100),1,0.8,(0,255,0),1)
        else:
            cv2.putText(im,target["label_str"][i], (boxes[i][2]-60,boxes[i][3]-20),1,0.8,(0,255,0),1)
    

    # DisplayAllFaults(dl.masks)

    DisplayTargetMask(target,1)
    DisplayImg(im)

    
    # print(Labelstemp)
    # print(np.max(Classification))

    # Classification = Classification * (255 / np.max(Classification))
    # Classification = Classification.astype(np.uint8)

    # # Show image
    # cv2.imshow('Image', Classification)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()