#!/usr/bin/python3

import scipy.io as sci
import cv2
import glob
import numpy as np
import pandas as pd
# from fastai.vision.all import *
import torch

class solar_panel_data():

    def __init__(self, img_dir, gt_dir, transforms = None, filter = True, area_limit=100):
        self.ImageDir = img_dir + "*"
        self.GTDir = gt_dir + "*"

        self.transforms = transforms

        self.files, self.masks = self.Load()

        if filter:
            print("Removing files without labels from path...")
            print(f'Avaliable files: {len(self.files)}')
            self.RemoveNoLabels()
            print(f'Avaliable files: {len(self.files)}')
            print(f"Removing files where all areas are smaller than {area_limit} square pixel...")
            self.RemoveBadArea(area_limit)
            print(f'Avaliable files: {len(self.files)}')
        


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

    def CleanErrors(self, filename, mask):
        # print(filename[-27:])
        if filename[-27:] == "10_4081_Cell_Row5_Col_3.png":
            mask[114:119, 264:277] = 0
            print("Found bad mask. Doing cleanup...")
        
        return mask


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

    def RemoveNoLabels(self):
        names_m = []
        flag = False
        for i in range(len(self.masks)):
            GT = sci.loadmat(self.masks[i])
            Labelstemp = GT['GTLabel'] # fault labels

            # if self.masks[i][-27:] == "10_4081_Cell_Row5_Col_4.mat":
            #     print("Remove bad measurement")
            #     flag = True

            if Labelstemp.size > 0 and not flag:
                names_m.append(self.masks[i])
            
            flag = False
        
        self.masks = names_m
        n_f = len(self.ImageDir)-1
        n_m = len(self.GTDir)-1
        self.files = self.RemoveNoMatches(self.files,names_m, n_f, n_m)

    def RemoveBadArea(self, area_limit):
        
        names_m = []
        for i in range(len(self.files)):
            if i % 50 == 0:
                print(".")
            if not self.__getitem__(i, find_bad_area=True, area_limit=area_limit):
                names_m.append(self.masks[i])
        
        self.masks = names_m
        n_f = len(self.ImageDir)-1
        n_m = len(self.GTDir)-1
        self.files = self.RemoveNoMatches(self.files,names_m, n_f, n_m)

    def PrintPaths(self):
        for i in range(len(self.files)):
            print(self.files[i])
            print(self.masks[i])
            print("------------------------")

    # def get_y(self):
    #     # Load .mat file
    #     GT = sci.loadmat(path)
    #     mask = GT['GTMask'] # fault mask
    #     return mask

    # def __getDatablock__(self):
    #     codes = ["No Failure", "Crack A", "Crack B", "Crack C", "Finger Failure"]
    #     return DataBlock(blocks = (ImageBlock, MaskBlock(codes=codes)),
    #                      get_items = get_image_files,
    #                      get_y = self.get_y,
    #                      splitter  = RandomSplitter(),
    #                      )

    def __getitem__(self, idx, find_bad_area=False, area_limit=100):
        # load images and masks
        img_path = self.files[idx]
        mask_path = self.masks[idx]

        # Load .mat file
        GT = sci.loadmat(mask_path)
        Labelstemp = GT['GTLabel'] # fault labels
        Labels = np.transpose(Labelstemp)
        mask = GT['GTMask'] # fault mask
        # Load example image
        if not find_bad_area:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Check if this mask is found to be errenous
        mask = self.CleanErrors(img_path, mask)

        if Labels.size == 0:
            new_labels = []
            labels = []
            boxes = [[]]
            masks = mask

        else:
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

            if len(obj_ids) is not len(new_labels):
                # print(obj_ids)
                # print(new_labels)
                # print(f'{len(obj_ids)} unique objects and {len(new_labels)} labels')
                # print(f'Discard: {img_path}')
                return True

            # print(Labels)
            # print(obj_ids)
            # print(len(masks))
            
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

        if len(boxes) is not len(obj_ids):
                # print(f'{len(obj_ids)} unique objects and {len(boxes)} bounding boxes')
                # print(f'Discard: {img_path}')
                return True

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # print(f'Area before check: {area}')
        # print(f'Labels: {new_labels}')
        # print(f'Length of labels: {len(new_labels)}')
        # print(f'Length of masks: {len(masks)}')

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
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])

        if len(new_labels) > 0:
            if find_bad_area:
                return False
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes = torch.as_tensor([[]], dtype=torch.float32)
            area = []

            if find_bad_area:
                return True

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["label_str"] = new_labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

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
        label_dic = {
            'Crack A': 1,
            'Crack B': 2,
            'Crack C': 3,
            'Finger Failure': 4
        }

        lab = []
        for i in range(len(labels)):
            lab.append(label_dic[labels[i]])

        return lab


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
    ImageDir = "../../data/Serie1_CellsAndGT/CellsCorr/"
    GTDir = "../../data/Serie1_CellsAndGT/MaskGT/"
    # GTDir = "../../data/Serie1_CellsAndGT/mask/"

    dl = solar_panel_data(ImageDir, GTDir, filter = True)

    # dl.PrintPaths()

    # num = 8499#8500#8512#8511#8697 #4494
    num = 11
    im, target = dl.__getitem__(num)

    print(f'Fault string: {target["label_str"]}')
    print(f'Fault label: {target["labels"].numpy()}')

    boxes = target["boxes"].numpy().astype(np.uint32)
    print(f'Boxes: {boxes}')

    print(f'Area: {target["area"]}')
    print(f'Number of masks: {len(target["masks"])}')

    if (len(boxes[0]) > 0):
        for i in range(len(boxes)):
            cv2.rectangle(im,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,255,0),2)

            xc = ((boxes[i][2]/2+boxes[i][0]/2))
            if np.abs(xc) > np.abs(xc-im.shape[0]):
                xc = (xc-50).astype(np.uint64)
            else:
                xc = (xc+15).astype(np.uint64)
            yc = ((boxes[i][3]/2+boxes[i][1]/2)).astype(np.uint64)
            print(f'(xc,yc) = ({xc},{yc})')
            
            try:
                cv2.putText(im,target["label_str"][i], (xc,yc),1,0.8,(0,255,0),1)
            except:
                print(f'Length of label_str: {len(target["label_str"])}')
            
            DisplayTargetMask(target,i)

    # DisplayAllFaults(dl.masks)



    DisplayImg(im)


if __name__ == '__main__':
    main()