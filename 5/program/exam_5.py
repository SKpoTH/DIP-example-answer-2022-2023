import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.measure import regionprops

def circularity(area, perimeter):
    '''
        Find Circularity
    '''
    circular = 4 * np.pi * (area / (perimeter**2))

    return circular

def classifier(feature_vect):
    '''
        Create Condition Classifer
    '''
    # -> Find Circularity
    circular = circularity(feature_vect.area, feature_vect.perimeter)

    ### -> Classification Condition
    if circular >= 0.75:
        if feature_vect.intensity_mean[1] >= 150:
            c = "Apple"
        else:
            c = "Tomato"
    else:
        if feature_vect.intensity_mean[1] >= 150:
            c = "Banana"
        else:
            c = "Pitaya"

    return c

INPUT_PATH = "../images/input/"
SEGMENT_PATH = "../images/segment/"

if __name__ == "__main__":
    # >>> Get image files
    input_files = glob(INPUT_PATH + "*")
    seg_files = glob(SEGMENT_PATH + "*")
    
    for i in range(len(input_files)):
        ### >>> Read images
        # Input Image
        input_img = cv.imread(input_files[i])
        input_img = cv.cvtColor(input_img, cv.COLOR_BGR2RGB)
        # Segment Image
        seg_img = cv.imread(seg_files[i])
        seg_img = cv.cvtColor(seg_img, cv.COLOR_BGR2GRAY)
        seg_img = np.where(seg_img > 127, 1, 0).astype(np.uint8)
        # Masking Image
        mask_img = cv.bitwise_and(input_img, input_img, mask=seg_img)

        ### -> Object Analyze
        _, label_img = cv.connectedComponents(seg_img)
        feature_vect = regionprops(label_img, input_img)
        for i in range(len(feature_vect)):
            c = classifier(feature_vect[i])
        
        ### => Answer Fruit Class
        print(f"Fruit Class: {c}")

        # ~> Display
        plt.subplot(1, 2, 1)
        plt.imshow(input_img, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow(mask_img, cmap="gray")
        plt.show()