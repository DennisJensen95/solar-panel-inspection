from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


def transform_torch_to_cv2(image, channels=3):
    transform = transforms.ToPILImage()

    image = np.array(transform(image))
    # print(np.shape(image))
    image = np.reshape(image, (224, 224, channels))

    return image


def transform_image(image, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    # image.to(device)
    return [image]


def inv_normalize(img):
    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.225)
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1/norm_std[0], 1/norm_std[1], 1/norm_std[2]]),
                                   transforms.Normalize(mean=[-norm_mean[0], -norm_mean[1], -norm_mean[2]],
                                                        std=[1., 1., 1.]),
                                   ])
    return invTrans(img)
