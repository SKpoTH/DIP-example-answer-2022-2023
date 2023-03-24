import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology as skmorph
from skimage.segmentation import flood_fill
from glob import glob

def euclideanDistance(input_rec, center_pos):
    '''
        Euclidean Distances
    '''
    ### -> Find Euclidean Distances
    distances = np.sqrt(np.sum((input_rec - center_pos)**2, axis=1))

    return distances

def mahalanobisDistance(input_rec, center_pos):
    '''
        Mahalanobis Distances
    '''
    ### -> Find Invert Covariance
    cov = np.cov(input_rec.T)
    inv_cov = np.linalg.inv(cov)
    ### -> Find Mahalanobis Distance
    delta = input_rec - center_pos
    distances = np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))

    return distances

def colorRangeDistance(input_img, center_pos, cutoff, distance_type="euclidean"):
    '''
    '''
    y, x, z = input_img.shape

    ### -> Flattern into Records
    input_rec = input_img.reshape(y*x, z)
    center_pos = center_pos * np.ones((y*x, 1))
    ### -> Find Distance
    if distance_type == "euclidean":
        distances = euclideanDistance(input_rec, center_pos)
    elif distance_type == "mahalanobis":
        distances = mahalanobisDistance(input_rec, center_pos)
    ### -> Cutoff Color
    output_rec = np.where(distances > cutoff, 0, 1)
    ### -> Reshape into image
    output_img = output_rec.reshape((y, x))

    return output_img

def floodFill(input_img):
    '''
        Fill Holes
    '''
    hole_img = input_img.copy()

    # -> Flood Fill Mask Temp
    mask = np.zeros((input_img.shape[0]+2, input_img.shape[1]+2), np.uint8)
    mask[1:-1,1:-1] = hole_img
    # -> Flood Fill
    hole_img = flood_fill(mask, (0, 0), 1)[1:-1,1:-1]
    # cv.floodFill(hole_img, mask, (0, 0), 1)
    hole_img = np.logical_not(hole_img)
    # -> Merge Original and Filled Area
    output_img = np.logical_or(input_img, hole_img)

    return output_img.astype(np.uint8)

def findIoU(input_img, gt_img):
    '''
        Find IoU
    '''
    intersect = np.logical_and(input_img, gt_img)
    union = np.logical_or(input_img, gt_img)
    # Intersection Over Union
    iou = np.sum(intersect) / np.sum(union)

    return iou

INPUT_PATH = "../images/input/"
GT_PATH = "../images/segment/"

if __name__ == "__main__":
    # >>> Get input image files
    input_files = glob(INPUT_PATH + '*')
    gt_files = glob(GT_PATH + '*')

    for i in range(len(input_files)):
        ### >>> Read Image Files
        # Input Image
        input_img = cv.imread(input_files[i])
        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        # Groundtruth Image
        gt_img = cv.imread(gt_files[i])
        gt_img = cv.cvtColor(gt_img, cv.COLOR_BGR2GRAY)
        gt_img = np.where(gt_img > 127, 1, 0)

        ### -> Preprocessing
        ench_img = cv.cvtColor(input_img, cv.COLOR_RGB2HSV)
        ench_img[:,:,2] = cv.equalizeHist(ench_img[:,:,2])
        ench_img = cv.cvtColor(ench_img, cv.COLOR_HSV2RGB)
        # ench_img = input_img

        ### -> Red Color Segmentation
        tomato_color = (255, 0, 0)
        cutoff = 150
        seg_img = colorRangeDistance(ench_img, tomato_color, cutoff, distance_type="euclidean")

        ### -> Postprocessing
        # - Closing
        stre = skmorph.disk(3)
        morph_img = cv.morphologyEx(np.uint8(seg_img), cv.MORPH_OPEN, stre)
        morph_img = cv.morphologyEx(np.uint8(morph_img), cv.MORPH_CLOSE, stre)
        # - Flood Fill
        output_img = floodFill(morph_img)

        ### -> IoU
        print(f"IoU: {findIoU(output_img, gt_img)}")

        # ~> Display Images
        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.subplot(1, 2, 2)
        # plt.imshow(ench_img)
        # plt.imshow(seg_img, cmap="gray")
        plt.imshow(output_img, cmap="gray")
        plt.show()

