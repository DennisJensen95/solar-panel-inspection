from ..data_load.data_load import inv_normalize, transform_torch_to_cv2
from torchvision import transforms
import numpy as np
import copy
import cv2


def visualize_faults(im, target_pred, inv_norm=True):

    if inv_norm:
        im = inv_normalize(im)

    im = transform_torch_to_cv2(im)

    boxes_pred = target_pred["boxes"].detach().numpy().astype(np.uint32)
    masks_pred = target_pred["masks"].detach().numpy()

    masks_pred = np.reshape(masks_pred, (len(masks_pred), 224, 224))

    print(np.shape(masks_pred))

    image = copy.copy(im)
    if len(masks_pred) == 0:
        print("No predictions")
        return

    if len(masks_pred) > 0:
        for i, mask_pred in enumerate(masks_pred):
            xc = boxes_pred[i][2] / 2 + boxes_pred[i][0] / 2
            if np.abs(xc) > np.abs(xc - im.shape[0]):
                xc = (xc - 50).astype(np.uint64)
            else:
                xc = (xc + 25).astype(np.uint64)
            yc = ((boxes_pred[i][3] / 2 + boxes_pred[i]
                   [1] / 2)).astype(np.uint64)

            try:
                cv2.putText(
                    im, str(target_pred["labels"][i].numpy()
                            ), (xc, yc), 1, 0.8, (0, 255, 0), 1
                )
            except:
                print("Cannot print labels")

            overlay_pred = np.zeros(im.shape, im.dtype)
            overlay_pred[:, :] = (0, 255, 0)

            mask_pred_copy = cv2.bitwise_and(
                overlay_pred, overlay_pred, mask=mask_pred)
            im = cv2.addWeighted(mask_pred_copy, 0.2, im, 0.8, 0)

        cv2.imshow("Image", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
