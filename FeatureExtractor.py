#!/usr/bin/env python
# coding: utf-8

import os
import glob
import shutil
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image
from skimage.feature import hog
from skimage.feature import haar_like_feature
from skimage.feature import local_binary_pattern as lbp
from skimage.io import imread
from skimage.transform import integral_image
from sklearn.externals import joblib

'''
class GeneralImageProcess is used for general 
image process that you can apply it to your 
project many times. 
'''
class GeneralImageProcess():
    def __init__(self, config):
        self.config = config

    def resize_crop_by_short(self, img, short_len=64):
        '''
        This function is used to resize and crop the input image
        to shape(short_len, short_len), first this function resizes
        the image in order to make the short length of the resized
        image equal to short_len, then the longer side of the image
        is cropped so that the shape of the final image is (short_len,
        short_len)
        :param img: the input image
        :param short_len: the length of the image after
                        resizing and cropping.
        :return: image of shape(short_len, short_len)
        '''
        (x, y) = img.size
        if x > y:
            y_s = short_len
            x_s = int(x * y_s / y)
            x_l = int(x_s / 2) - int(short_len / 2)
            x_r = int(x_s / 2) + int(short_len / 2)
            img = img.resize((x_s, y_s))
            box = (x_l, 0, x_r, short_len)
            img = img.crop(box)
        else:
            x_s = short_len
            y_s = int(y * x_s / x)
            y_l = int(y_s / 2) - int(short_len / 2)
            y_r = int(y_s / 2) + int(short_len / 2)
            img = img.resize((x_s, y_s))
            box = (0, y_l, short_len, y_r)
            img = img.crop(box)
        return img

    def resize_by_short(self, img, short_len=256):
        '''
        This function is used to resize the image according to
        the short side of the image, the output image shape
        (short_len, y*short_len/x)(if x < y) or
        (x*short_len/y,short_len)(if x > y)
        :param img : The input image
        :param short_len : The length of the short side of the
        resized image
        :return : The resized image
        '''
        print(img.size)
        (x, y) = img.size
        if x > y:
            y_s = short_len
            x_s = int(x * y_s / y)
            img = img.resize((x_s, y_s))
        else:
            x_s = short_len
            y_s = int(y * x_s / x)
            img = img.resize((x_s, y_s))
        return img

    def image_preprocess_size(self, short_len=64):
        '''
        This function is used to resize and crop all images and
        save the final images shape(short_len, short_len)
        :param short_len: the length of the resized and cropped image
        :return: None
        '''
        pPath = self.config.DIR_PATHS["POS_IMG_PH"] + "/"
        nPath = self.config.DIR_PATHS["NEG_IMG_PH"] + "/"
        pImages = [pPath + x for x in os.listdir(pPath) if not x.startswith('.')]
        nImages = [nPath + x for x in os.listdir(nPath) if not x.startswith('.')]

        sizes_pos = []
        sizes_neg = []
        print("==> Resize and crop positive images")
        for img_name in tqdm(pImages):
            img = Image.open(img_name)
            sizes_pos.append(img.size)
            img = self.resize_crop_by_short(img, short_len)
            img.save(pPath + os.path.split(img_name)[1])

        print("==> Resize and crop negtive images")
        for img_name in tqdm(nImages):
            img = Image.open(img_name)
            sizes_neg.append(img.size)
            img = self.resize_crop_by_short(img, short_len)
            img.save(nPath + os.path.split(img_name)[1])

    def sliding_window(self, image, window_size, step_size):
        """
            This function returns a patch of the input image `image` of size equal
            to `window_size`. The first image returned top-left co-ordinates (0, 0)
            and are increment in both x and y directions by the `step_size` supplied.
            So, the input parameters are -
            * `image` - Input Image
            * `window_size` - Size of Sliding Window
            * `step_size` - Incremented Size of Window

            The function returns a tuple -
            (x, y, im_window)
            where
            * x is the top-left x co-ordinate
            * y is the top-left y co-ordinate
            * im_window is the sliding window image
        """
        for y in range(0, image.shape[0], step_size[1]):
            for x in range(0, image.shape[1], step_size[0]):
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

    def overlapping_area(self, detection_1, detection_2):
        """
            Function to calculate overlapping area'si
            `detection_1` and `detection_2` are 2 detections whose area
            of overlap needs to be found out.
            Each detection is list in the format ->
            [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
            The function returns a value between 0 and 1,
            which represents the area of overlap.
            0 is no overlap and 1 is complete overlap.
            Area calculated from ->
            http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        """
        # Calculate the x-y co-ordinates of the rectangles
        x1_tl = detection_1[0]
        x2_tl = detection_2[0]
        x1_br = detection_1[0] + detection_1[3]
        x2_br = detection_2[0] + detection_2[3]
        y1_tl = detection_1[1]
        y2_tl = detection_2[1]
        y1_br = detection_1[1] + detection_1[4]
        y2_br = detection_2[1] + detection_2[4]
        # Calculate the overlapping Area
        x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
        y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
        overlap_area = x_overlap * y_overlap
        area_1 = detection_1[3] * detection_2[4]
        area_2 = detection_2[3] * detection_2[4]
        total_area = area_1 + area_2 - overlap_area
        return overlap_area / float(total_area)

    def nms(self, detections, threshold=.5):
        """
            This function performs Non-Maxima Suppression.
            `detections` consists of a list of detections.
            Each detection is in the format ->
            [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection]
            If the area of overlap is greater than the `threshold`,
            the area with the lower confidence score is removed.
            The output is a list of detections.
        """
        if len(detections) == 0:
            return []
        # Sort the detections based on confidence score
        detections = sorted(detections, key=lambda detections: detections[2],
                            reverse=True)
        new_detections = []  # Unique detections will be appended to this list
        new_detections.append(detections[0])  # Append the first detection
        del detections[0]  # Remove the detection from the original list
        """
            For each detection, calculate the overlapping area
            and if area of overlap is less than the threshold set
            for the detections in `new_detections`, append the 
            detection to `new_detections`.
            In either case, remove the detection from `detections` list.
        """
        for index, detection in enumerate(detections):
            for new_detection in new_detections:
                if self.overlapping_area(detection, new_detection) > threshold:
                    del detections[index]
                    break
            else:
                new_detections.append(detection)
                del detections[index]
        return new_detections

'''
class Extractor is used to extract features from images and 
save the features as .feat files so that the features files 
can be loaded when needed.
'''
class Extractor(GeneralImageProcess):
    def __init__(self, config):
        super(Extractor, self).__init__(config)

    def process_image(self, image):
        '''
        This function is used to extract features from input image
        according to the config
        :param image: input image
        :return: The features extracted from input image
        '''
        if self.config.DES_TYPE == "HOG":
            fd = hog(image, block_norm='L2', pixels_per_cell=self.config.PIXELS_PER_CELL)
        elif self.config.DES_TYPE == "LBP":
            fd = lbp(image, self.config.LBP_POINTS, self.config.LBP_RADIUS)
        elif self.config.DES_TYPE == "HAAR":
            fd = haar_like_feature(integral_image(image), 0, 0, 5, 5, 'type-3-x')
        elif self.config.DES_TYPE == "ORB":
            Orb = cv2.ORB_create(nfeatures=self.config.ORBParam["Nfeatures"],
                                edgeThreshold=self.config.ORBParam["Edgethresh"])
            fd = Orb.detectAndCompute(image, None)[1] # Compute the description of the keypoints
        else:
            raise KeyError("==> The Processing method does not exist!")
        return fd

    def get_kmeans_features(self):
        '''
        This function is used to extract point features with k-means,
        the input is a key point description vector, and k-means gives
        it a cluster label range(0, n_clusters)
        :return: None
        '''
        if self.config.DES_TYPE != "ORB":
            raise Exception("Can not extract feature, beacause %s does not support."\
                            % self.config.DES_TYPE)

        PosSamples = []
        NegSamples = []
        ## load the k-means model
        if not self.config.USE_MINIBATCH:
            km = joblib.load(os.path.join(self.config.DIR_PATHS["MODEL_DIR_PH"], "_kmeans.pkl"))
        else:
            km = joblib.load(os.path.join(self.config.DIR_PATHS["MODEL_DIR_PH"], "_minibatchkmeans.pkl"))

        ## get points features with k-means
        print("==> Get positive images points features......")
        for feat_path in tqdm(glob.glob(os.path.join(self.config.DIR_PATHS["POS_FEAT_PH"], "*.feat"))):
            feat = joblib.load(feat_path)
            if feat is None:
                PosSamples.append(np.array([]))
                continue
            feat = km.predict(feat)
            PosSamples.append(feat)
        print("==> Get negtive images points features......")
        for feat_path in tqdm(glob.glob(os.path.join(self.config.DIR_PATHS["NEG_FEAT_PH"], "*.feat"))):
            feat = joblib.load(feat_path)
            if feat is None:
                NegSamples.append(np.array([]))
                continue
            feat = km.predict(feat)
            NegSamples.append(feat)

        ## Save the points features
        print("==> Saving points features......")
        joblib.dump(PosSamples, os.path.join(self.config.DIR_PATHS["POS_FEAT_PH"], "_pos_points_features.pkl"))
        joblib.dump(NegSamples, os.path.join(self.config.DIR_PATHS["NEG_FEAT_PH"], "_neg_points_features.pkl"))

    def extract_features(self):
        '''
        This function is used to extract the features of the pos images
        and neg images, and save the extracted features as .feat files.
        :return: None
        '''
        if os.path.exists(self.config.DIR_PATHS["POS_FEAT_PH"]):
            shutil.rmtree(self.config.DIR_PATHS["POS_FEAT_PH"])
        if os.path.exists(self.config.DIR_PATHS["NEG_FEAT_PH"]):
            shutil.rmtree(self.config.DIR_PATHS["NEG_FEAT_PH"])
        os.makedirs(self.config.DIR_PATHS["POS_FEAT_PH"])
        os.makedirs(self.config.DIR_PATHS["NEG_FEAT_PH"])

        print("==> Calculating the descriptors for the positive samples and saving them")
        for im_path in tqdm(glob.glob(os.path.join(self.config.DIR_PATHS["POS_IMG_PH"], "*"))):
            if self.config.DES_TYPE == "ORB":
                im = cv2.imread(im_path, 0)
            else:
                im = imread(im_path, as_grey=True)
            fd = self.process_image(im)
            if fd is None: continue
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(self.config.DIR_PATHS["POS_FEAT_PH"], fd_name)
            joblib.dump(fd, fd_path)
        print("==> Positive features saved in {}".format(self.config.DIR_PATHS["POS_FEAT_PH"]))

        print("==> Calculating the descriptors for the negative samples and saving them")
        for im_path in tqdm(glob.glob(os.path.join(self.config.DIR_PATHS["NEG_IMG_PH"], "*"))):
            if self.config.DES_TYPE == "ORB":
                im = cv2.imread(im_path, 0)
            else:
                im = imread(im_path, as_grey=True)
            fd = self.process_image(im)
            if fd is None: continue
            fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
            fd_path = os.path.join(self.config.DIR_PATHS["NEG_FEAT_PH"], fd_name)
            joblib.dump(fd, fd_path)
        print("==> Negative features saved in {}".format(self.config.DIR_PATHS["NEG_FEAT_PH"]))
        print("==> Completed calculating features from training images")
