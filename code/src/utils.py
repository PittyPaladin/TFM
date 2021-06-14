import os
import json
import cv2
import numpy as np
from skimage import io, exposure, feature, color, transform, filters
from sklearn import svm
import matplotlib.pyplot as plt


def sliding_window(image, window_shape):
    # step depends on the shape of the window
    step = (int(window_shape[0]*0.5), int(window_shape[1]*0.5))
    for i in range(0, image.shape[0], step[0]):
        for j in range(0, image.shape[1], step[1]):
            # horizontal overflow of the window over the image
            if i + window_shape[0] > image.shape[0] and not j + window_shape[1] > image.shape[1]:
                row = image.shape[0] - window_shape[0]
                col = j
                patch = image[row:, col:col + window_shape[1]]
                yield row, col, patch
            # vertical overflow of the window over the image
            elif not i + window_shape[0] > image.shape[0] and j + window_shape[1] > image.shape[1]:
                row = i
                col = image.shape[1] - window_shape[1]
                patch = image[row:row + window_shape[0], col:]
                yield row, col, patch
            # horizontal and vertical overflow of the window over the image
            elif i + window_shape[0] > image.shape[0] and j + window_shape[1] > image.shape[1]:
                row = image.shape[0] - window_shape[0]
                col = image.shape[1] - window_shape[1]
                patch = image[row:, col:]
                yield row, col, patch
            # regular case
            else:
                row = i
                col = j
                patch = image[row:row + window_shape[0], col:col + window_shape[1]]
                yield  row, col, patch

def image_pyramid(image, window_shape, imageVSwindow_height_ratio=0.2, fx=2/3, fy=2/3, interpolation=cv2.INTER_CUBIC):
    # resize so that the window's height is X% the image's height
    ar = image.shape[1] / image.shape[0] # aspect ratio
    newH = int(window_shape[0] / imageVSwindow_height_ratio)
    newW = int(ar * newH)
    scaleH = newH / image.shape[0]
    scaleW = newW / image.shape[1]
    res = cv2.resize(image, (newW, newH), interpolation=interpolation)
    yield scaleH, scaleW, res

    nrescales = 1
    while True:
        # rescale again
        res = cv2.resize(res, None, fx=fx, fy=fy, interpolation=interpolation)
        # break if height or width of the rescaled image is smaller than the one of the sliding window
        if window_shape[0] > res.shape[0] or window_shape[1] > res.shape[1]:
            break
        yield scaleH*fx**nrescales, scaleW*fy**nrescales, res
        nrescales += 1

def compute_hog(image, window_shape):
    if len(image.shape) == 2: # it's already grayscale
        gray = image
    elif image.shape[2] == 4:
        gray = color.rgb2gray(color.rgba2rgb(image))
    elif image.shape[2] == 3:
        gray = color.rgb2gray(image)
    # blur
    gray = filters.gaussian(gray, sigma=2)
    # resize to a fixed size
    gray = transform.resize(gray, window_shape)
    # compute HoG
    (H, hog_image) = feature.hog(gray, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", 
        feature_vector=True, visualize=True)
    return (H, hog_image)


def labelIntegrityCheck(imWidth, imHeight, filename, label_dict):
    """
    Check if the dictionary of labels loaded from the labels json file has incoherent information.

    Args:
        imWidth (int): width of the image in pixels.
        imHeight (int): height of the image in pixels.
        filename (str): name of the image's filename with the extension.
        label_dict (dict): dictionary of labels loaded from the json labels file.

    Raises:
        ValueError: when some information is not coherent or missing.
    """
    # check filename in the labels dict matches the actual image filename
    if not label_dict['image']['file_name'] == filename:
        raise ValueError("Image filename reported in labels file does not match actual image filename.")

    # check image shape in the labels dict matches the actual image shape
    if not label_dict['image']['height'] == imHeight or not label_dict['image']['width'] == imWidth:
        raise ValueError("Image height and/or width reported in labels file does not match the one from actual image.")

    # check annotations make sense
    for annot in label_dict['annotation']:
        for key, value in annot.items():
            if value is None:
                raise ValueError("Content of the labels file is corrupt.")

def read_image_labels(image_path): # TODO: change function name to `read_corresponding_labels`
    # open image
    image = io.imread(image_path)
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_name = image_path.rsplit("/")[-1]
    # open labels file
    label_path = image_path.rsplit(".")[0]
    label_path += '_MASK.json'
    if os.path.exists(label_path):
        with open(label_path, "r") as rf:
            try:
                labels_data = json.load(rf)
                labelIntegrityCheck(image_width, image_height, image_name, labels_data)
                return labels_data
            except ValueError as error:
                print(error)
            except:
                print("Unexpected error!")
    else:
        print(f"No labels file for {image_path}!")

def labels2bboxes(labels_dict):
    """Takes the dictionary of labels and returns a list of class name and bounding 
    box coordinates as (class_name, [x, y, width, height])."""
    bboxes_list = [
        (annot["name"], annot["bbox"])
        for annot in labels_dict["annotation"]
    ]
    return bboxes_list        

def show_bboxes(image_path, bboxes_list):
    """Put all bounding boxes in the image and display it."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots()
    plt.imshow(io.imread(image_path))
    for bbox in bboxes_list:
        bbox_classname = bbox[0]
        top_x = bbox[1][0]
        top_y = bbox[1][1]
        width = bbox[1][2]
        height = bbox[1][3]
        ax.add_patch(FancyBboxPatch([top_x, top_y], width, height, fill=False, edgecolor="m"))
        ax.text(top_x, top_y + int(0.05*height), bbox_classname, color="m")
    plt.show()

def intersection_over_union(bboxA, bboxB):
    """Compute the intersection over the union of two bounding boxes. If they 
    are not the same class automatically return 0."""
    # check the boxes are the same class
    if bboxA[0] != bboxB[0]:
        return 0
    # compute area of the intersection
    intersect_xmin = max(bboxA[1][0], bboxB[1][0])
    intersect_ymin = max(bboxA[1][1], bboxB[1][1])
    intersect_xmax = min(bboxA[1][0] + bboxA[1][2], bboxB[1][0] + bboxB[1][2])
    intersect_ymax = min(bboxA[1][1] + bboxA[1][3], bboxB[1][1] + bboxB[1][3])
    intersection_area = max(0, intersect_xmax - intersect_xmin) * max(0, intersect_ymax - intersect_ymin)
    # areas of the bounding boxes
    bboxA_area = bboxA[1][2] * bboxA[1][3]
    bboxB_area = bboxB[1][2] * bboxB[1][3]
    iou = intersection_area / (bboxA_area + bboxB_area - intersection_area)
    return iou