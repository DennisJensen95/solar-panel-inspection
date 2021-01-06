import scipy.io as sci
import cv2
import glob
import numpy as np


def main():
    # Generate paths to all viable images
    files, masks = GeneratePath()

    # Print example
    imgNum = 352
    print(files[imgNum])
    print(masks[imgNum])

    # Load .mat file
    GT = sci.loadmat(masks[imgNum])
    Labelstemp = GT['GTLabel']  # fault labels
    Labels = np.transpose(Labelstemp)
    Classification = GT['GTMask']  # fault mask

    # Load example image
    Image = cv2.imread(files[imgNum], cv2.IMREAD_GRAYSCALE)

    # Show image
    cv2.imshow('Image', Image)
    cv2.waitKey()


def GeneratePath():
    """Generates path to images and removes 
    any that does not have a corresponding mask 

    Returns:
        [list]: [path to images]
        [list]: [path to masks/labels]
    """

    # Path to directories
    ImageDir = "data/Serie1_CellsAndGT/CellsCorr/*.png"
    GTDir = "data/Serie1_CellsAndGT/MaskGT/*"

    # Load paths using glob
    files = sorted(glob.glob(ImageDir))
    masks = sorted(glob.glob(GTDir))

    files = RemoveNoMatches(files, masks)

    return files, masks


def RemoveNoMatches(a, b):
    """Returns any path that does not have a corresponding mask 

    Returns:
        [list]: [image paths with corresponding mask]
    """

    # Beware of the index!!
    names = [w[52:-4] for w in a]  # 33 + 19 = 52
    names_m = [w[48:-4] for w in b]  # 30 + 18 = 48

    # Only keep strings that occur in each list
    names = [x for x in names if x in names_m]

    # Add the path back and return
    return [a[0][:52] + w + a[0][-4:] for w in names]


if __name__ == '__main__':
    main()
