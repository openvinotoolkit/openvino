# Use case: user wants to run the inference on set of pictures
# and store the results of the inference (e.g. in a database)
# The Inference Queue allows him to run inference in parallel jobs.

import numpy as np
import os
import time
from openvino.inference_engine import IECore
from openvino.inference_engine import TensorDesc
from openvino.inference_engine import Blob
from openvino.inference_engine import StatusCode
from openvino.inference_engine import InferQueue


def image_path(name):
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, 'validation_set', '224x224', name)
    return path_to_img


def read_image(name):
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(image_path(name))
    if image is None:
        raise FileNotFoundError('Input image not found')

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image.reshape((n, c, h, w))
    return image


def get_images():
    images = []

    images += [read_image('dog.bmp')]
    images += [read_image('cat1.bmp')]
    images += [read_image('cat2.bmp')]
    images += [read_image('dog1.bmp')]
    images += [read_image('dog2.bmp')]
    images += [read_image('dog3.bmp')]

    return images
