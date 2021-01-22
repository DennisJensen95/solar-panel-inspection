import torchvision.transforms.functional as TF
from torchvision import transforms
import scipy.io as sci
from PIL import Image
import glob
import os

class LoadImages():
    
    def __init__(self, img_dir, mask_dir, normalize=False):
        self.ImageDir = img_dir + "*"
        self.GTDir = mask_dir + "*"
        self.normalize = normalize
        self.files, self.masks = self.GeneratePath()
        self.files_fault, self.masks = self.RemoveNoLabels()
        self.norm_mean = (0.485, 0.456, 0.406) 
        self.norm_std = (0.229, 0.224, 0.225)
        
        self.errors = 0 
        
        for file in self.files:
            self.errors += self.compare_image_error_file_path(file)
            
        print(f'Number of error files: {self.errors}')
    
    def compare_image_error_file_path(self, img_file):
        if img_file in self.files_fault:
            return 1
        else:
            return 0
        
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

        files = self.RemoveNoMatches(files, masks)
        
        return files, masks

    def CleanErrors(self, filename, mask):
        """Special case where data can be fixed by simply removing some numbers

        Args:
            filename (string): the image path of the file in question
            mask (numpy array): the given mask

        Returns:
            [numpy array]: fixed mask
        """
        if filename[-27:] == "10_4081_Cell_Row5_Col_3.png":
            mask[114:119, 264:277] = 0

        return mask

    def RemoveNoMatches(self, files, masks):
        """Returns any path that does not have a corresponding mask

        Returns:
            [list]: [image paths with corresponding mask]
        """
        matches = []
        file_list = [file.replace("Corr", "").replace(".png", "") for file in files] 
        masks_list = [os.path.basename(os.path.normpath(mask)).replace("GT_", "").replace(".mat", "") for mask in masks]
        for i, file in enumerate(files):
            file_name = os.path.basename(os.path.normpath(file_list[i]))
            
            if file_name in masks_list:
                matches.append(file)
                
        return matches

    def RemoveNoLabels(self):
        """Removes measurements that does not contain a label"""

        names_m = []
        flag = False
        for i in range(len(self.masks)):
            GT = sci.loadmat(self.masks[i])
            Labelstemp = GT["GTLabel"]  # fault labels

            # Make a list of all files that has a label
            if Labelstemp.size > 0 and not flag:
                names_m.append(self.masks[i])

            flag = False

        # Update mask list
        masks = names_m
        n_f = len(self.ImageDir) - 1
        n_m = len(self.GTDir) - 1

        # Update files list
        files_fault = self.RemoveNoMatches(self.files, names_m)
        
        return files_fault, masks
    
    def transform_image(self, image):
        resize = transforms.Resize(size=(224, 224))
        norm = transforms.Normalize(self.norm_mean, self.norm_std)
        image = TF.to_tensor(image)
        image = resize(image)
        if self.normalize:
            image = norm(image)
        
        return image
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        label = self.compare_image_error_file_path(filename)
        
        img = Image.open(filename)
        img = img.convert('RGB')
        
        img = self.transform_image(img)
        
        return img, label
    
    def __len__(self):
        return len(self.files)