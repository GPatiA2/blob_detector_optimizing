import cv2
import numpy as np
import os
import argparse
import json
from copy import deepcopy

class TraditionalClassifier():

    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        self.detector = cv2.SimpleBlobDetector_create(params)
        
    def set_params_dict(self, params, dict_params):

        params.minThreshold = dict_params['minThreshold'] if 'minThreshold' in dict_params.keys() else params.minThreshold
        params.maxThreshold = dict_params['maxThreshold'] if 'maxThreshold' in dict_params.keys() else params.maxThreshold
        params.thresholdStep = dict_params['thresholdStep'] if 'thresholdStep' in dict_params.keys() else params.thresholdStep

        params.filterByArea = dict_params['filterByArea'] if 'filterByArea' in dict_params.keys() else params.filterByArea
        params.minArea = dict_params['minArea'] if 'minArea' in dict_params.keys() else params.minArea 
        params.maxArea = dict_params['maxArea'] if 'maxArea' in dict_params.keys() else params.maxArea

        params.filterByCircularity = dict_params['filterByCircularity'] if 'filterByCircularity' in dict_params.keys() else params.filterByCircularity
        params.minCircularity = dict_params['minCircularity'] if 'minCircularity' in dict_params.keys() else params.minCircularity
        params.maxCircularity = dict_params['maxCircularity'] if 'maxCircularity' in dict_params.keys() else params.maxCircularity

        params.filterByConvexity = dict_params['filterByConvexity'] if 'filterByConvexity' in dict_params.keys() else params.filterByConvexity
        params.minConvexity = dict_params['minConvexity'] if 'minConvexity' in dict_params.keys() else params.minConvexity
        params.maxConvexity = dict_params['maxConvexity'] if 'maxConvexity' in dict_params.keys() else params.maxConvexity

        params.filterByInertia = dict_params['filterByInertia'] if 'filterByInertia' in dict_params.keys() else params.filterByInertia
        params.minInertiaRatio = dict_params['minInertiaRatio'] if 'minInertiaRatio' in dict_params.keys() else params.minInertiaRatio
        params.maxInertiaRatio = dict_params['maxInertiaRatio'] if 'maxInertiaRatio' in dict_params.keys() else params.maxInertiaRatio

        params.minDistBetweenBlobs = dict_params['minDistBetweenBlobs'] if 'minDistBetweenBlobs' in dict_params.keys() else params.minDistBetweenBlobs

        self.detector = cv2.SimpleBlobDetector_create(params)

        return params
    
    def set_params(self, new_params):

        params = cv2.SimpleBlobDetector_Params()

        # params.minThreshold = new_params[0]
        # params.maxThreshold = new_params[1]
        # params.thresholdStep = new_params[2]

        # params.filterByArea = True
        # params.minArea = new_params[0]
        # params.maxArea = new_params[1]

        params.filterByCircularity = True
        params.minCircularity = new_params[0]
        params.maxCircularity = new_params[1]

        params.filterByConvexity = True
        params.minConvexity = new_params[2]
        params.maxConvexity = new_params[3]

        params.filterByInertia = True
        params.minInertiaRatio = new_params[4]
        params.maxInertiaRatio = new_params[5]

        # params.minDistBetweenBlobs = new_params[8]

        self.detector = cv2.SimpleBlobDetector_create(params)

    def get_blobs(self, im):

        keypoints = self.detector.detect(image = im)

        rgb_im = np.stack((im,)*3, axis=-1)

        im_with_keypoints = cv2.drawKeypoints(rgb_im, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return (im_with_keypoints, keypoints)
    
    def predict_batch(self, batch):

        batch_pred = []
        for im in batch:
            kp   = self.detector.detect(image = np.uint8(im.numpy()))
            pred = 1 if len(kp) > 0 else 0
            batch_pred.append(pred) 
        return batch_pred
    
    def predict_step(self, im):
        kp  = self.detector.detect(image = np.uint8(im))
        pred = 1 if len(kp) > 0 else 0
        return pred
    
    def get_max_params(self):
        
        max_params = []
        
        # threshold min & max
        # max_params.append(255)
        # max_params.append(255)

        # threshold step
        # max_params.append(200)

        # area min & max
        # max_params.append(1)
        # max_params.append(1000)

        # circularity min & max
        max_params.append(1)
        max_params.append(1)

        # convexity min & max
        max_params.append(1)
        max_params.append(1)

        # inertia min & max
        max_params.append(1)
        max_params.append(1)

        # min dist between blobs
        # max_params.append(5)

        return max_params

    def transforms(self):

        def preprocess(img):
            mean = np.mean(img[img > 0])
            std  = np.std(img[img > 0])
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img  = cv2.ximgproc.rollingGuidanceFilter(img, numOfIter= 10 )
            img  = cv2.threshold(img, mean + std, 255, cv2.THRESH_BINARY)[1]
            img  = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
            img  = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
            img  = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
            return img
        
        return preprocess
    

    
    
def load_dataset(im_path):

    dataset = []
    with open(os.path.join(im_path, 'tags.json'), 'r') as f:
        tags = json.load(f)

    tags_d = {}
    for it in tags:
        tags_d[it['name']] = it['label']

    for f in os.listdir(im_path):
        if f.endswith(".JPG"):
            t = 'bad' if tags_d[f] != 'healthy' else 'healthy'
            dataset.append((os.path.join(im_path, f), t))
        
    return dataset

def options():

    parser = argparse.ArgumentParser(description='Traditional Detector')
    parser.add_argument('--image_path', type=str, default='images', help='Path to dataset')
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = options()

    blob_params = {

        'minDistBetweenBlobs' : 1,

        'filterByArea' : False,
        'minArea' : 1,
        'maxArea' : 1000,

        'filterByCircularity' : False,
        'minCircularity' : 0.1,
        'maxCircularity' : 1,
        
        'filterByConvexity' : False,
        'minConvexity' : 0.1,
        'maxConvexity' : 1,
        
        'filterByInertia' : False,
        'minInertiaRatio' : 0.1,
        'maxInertiaRatio' : 1,
        
        # 'minThreshold' : 0,
        # 'maxThreshold' : 255,
        # 'thresholdStep' : 10
        'blobColor' : 255
    }


    detector = TraditionalClassifier()
    detector.set_params_dict(cv2.SimpleBlobDetector_Params(), blob_params)


    dataset = load_dataset(args.image_path)

    print("Dataset loaded")
    print(len(dataset))

    for im_path, tag in dataset:

        im = cv2.imread(im_path)
        im2 = detector.transforms()(deepcopy(im))
        im3 = detector.transforms()(deepcopy(im))

        im_with_keypoints, kp = detector.get_blobs(im2)

        pred = 'healthy' if len(kp) == 0 else 'bad'
        name = 'TAG = ' + tag + " PREDICTED = " + pred
        orig = "ORIGINAL WITH KEYPOINTS"

        print("Detected keypoints: ", len(kp))
        # cv2.namedWindow("CONTOURS DETECTED", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("CONTOURS DETECTED", 800, 800)
        # cnts = cv2.findContours(im3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # im3 = np.stack((im3,)*3, axis=-1)
        # im3 = cv2.drawContours(im3, cnts, -1, (0,255,0), 3)
        # cv2.imshow("CONTOURS DETECTED", im3)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 800, 800)
        cv2.namedWindow(orig, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(orig, 800, 800)
        cv2.imshow(name, im_with_keypoints)
        cv2.imshow(orig, cv2.drawKeypoints(im, kp, (0,255,0)))
        cv2.waitKey(0)

 