# -*- coding:utf-8 -*-

import glob
import os
import cv2
import numpy as np
import Config

from tqdm import tqdm
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from skimage.transform import pyramid_gaussian
from sklearn.externals import joblib

from FeatureExtractor import Extractor

## Initialize the Extractor
conf = Config.Config()
extractor = Extractor(conf)

class Classifier(object):
    def __init__(self, config):
        self.config = config

    def load_data(self):
        '''
        This function is used to load the training data, positive data
        and negtive data.The features of images are stored in self.fds,
        the labels are stored in self.labels.
        :return: None
        '''
        self.fds = []
        self.labels = []
        print("==> Loading the positive features")
        for feat_path in tqdm(glob.glob(os.path.join(self.config.DIR_PATHS["POS_FEAT_PH"], "*.feat"))):
            fd = joblib.load(feat_path)
            self.fds.append(fd.reshape(-1))
            self.labels.append(1)

        print("==> Load the negative features")
        for feat_path in tqdm(glob.glob(os.path.join(self.config.DIR_PATHS["NEG_FEAT_PH"], "*.feat"))):
            fd = joblib.load(feat_path)
            self.fds.append(fd.reshape(-1))
            self.labels.append(0)

    def train_classifier(self):
        '''
        This function is used to train the classifier, before this, you
        should prepare and load the image feature files, then inintialize
        the LinearSVC or other classification model to train.
        :return: None
        '''
        if self.config.CLF_TYPE is "LIN_SVM":
            clf = LinearSVC()
            print("==> Training a Linear SVM Classifier")
            clf.fit(self.fds, self.labels)
            joblib.dump(clf, self.config.MODEL_PH)
            print("==> Classifier saved to {}".format(self.config.MODEL_PH))
        elif self.config.CLF_TYPE is "MLP":
            clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(16, 32, 64), random_state=1)
            print("==> Training a Multi Layer Classifier")
            clf.fit(self.fds, self.labels)
            joblib.dump(clf, self.MODEL_PH)
            print("==> Classifier saved to {}".format(self.config.MODEL_PH))

    def load_model(self):
        '''
        This functiton is used to load the model
        :return: None
        '''
        self.clf = joblib.load(self.config.MODEL_PH)  # Load the classifier

    def predict(self, fd, score=False):
        '''
        This function is used to predict the class according to input fd
        :param fd: The input feature vector
        :param score: (bool)Ouput score or class, default class
        :return: class or score
        '''
        if self.config.CLF_TYPE == "LIN_SVM":
            if score:
                return self.clf.decision_function(fd)
            else:
                return self.clf.predict(fd)
        elif self.config.CLF_TYPE == "MLP":
            if score:
                return self.clf.predict_proba(fd)[0][1]
            else:
                return self.clf.predict(fd)

    def test_classifier(self):
        '''
        This function is used to test the classifier on test images,
        before this, you should run load_model() function to get the
        model.
        :return: None
        '''
        for im_path in [os.path.join(self.config.DIR_PATHS["TEST_IMG_DIR_PH"], i) for i in
                        os.listdir(self.config.DIR_PATHS["TEST_IMG_DIR_PH"]) if not i.startswith('.')]:
            # Read the Image
            im = Image.open(im_path).convert('L')
            im = np.array(extractor.resize_by_short(im))

            detections = []  # List to store the detections
            scale = 0  # The current scale of the image

            # Downscale the image and iterate
            for im_scaled in pyramid_gaussian(im, downscale=self.config.DOWNSCALE):
                cd = []  # This list contains detections at the current scale
                # If the width or height of the scaled image is less than
                # the width or height of the window, then end the iterations.
                if im_scaled.shape[0] < self.config.MIN_WDW_SIZE[1] or im_scaled.shape[1] < self.config.MIN_WDW_SIZE[0]:
                    break
                for (x, y, im_window) in extractor.sliding_window(im_scaled, self.config.MIN_WDW_SIZE, self.config.STEP_SIZE):
                    if im_window.shape[0] != self.config.MIN_WDW_SIZE[1] or im_window.shape[1] != self.config.MIN_WDW_SIZE[0]:
                        continue
                    # Calculate the HOG features
                    fd = extractor.process_image(im_window).reshape([1, -1])
                    pred = self.predict(fd, score=False)
                    if pred == 1:
                        if self.config.IF_PRINT: print("==> Detection:: Location -> ({}, {})".format(x, y))
                        if self.config.CLF_TYPE is "LIN_SVM":
                            if self.config.IF_PRINT: print(
                                "==> Scale ->  {} Confidence Score {} \n".format(scale, self.predict(fd, score=True)))
                            detections.append((x, y, self.predict(fd, score=True),
                                               int(self.config.MIN_WDW_SIZE[0] * (self.config.DOWNSCALE ** scale)),
                                               int(self.config.MIN_WDW_SIZE[1] * (self.config.DOWNSCALE ** scale))))
                        elif self.config.CLF_TYPE is "MLP":
                            if self.config.IF_PRINT: print("==> Scale ->  {} Confidence Score {} \n".format(scale,
                                                                                                     self.predict(fd, score=True)))
                            detections.append((x, y, self.predict(fd, score=True),
                                               int(self.config.MIN_WDW_SIZE[0] * (self.config.DOWNSCALE ** scale)),
                                               int(self.config.MIN_WDW_SIZE[1] * (self.config.DOWNSCALE ** scale))))
                        cd.append(detections[-1])

                    # If visualize is set to true, display the working of the sliding window
                    if self.config.VISUALIZE:
                        clone = im_scaled.copy()
                        for x1, y1, _, _, _ in cd:
                            # Draw the detections at this scale
                            cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                            im_window.shape[0]), (0, 0, 0), thickness=2)
                        cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                                                      im_window.shape[0]), (255, 255, 255), thickness=2)
                        cv2.imshow("Sliding Window in Progress", clone)
                        cv2.waitKey(30)

                # Move the the next scale
                scale += 1

            # Display the results before performing NMS
            clone = im.copy()

            # Draw the detections
            for (x_tl, y_tl, _, w, h) in detections:
                cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)

            detections_fin = extractor.nms(detections, self.config.THRESHOLD)  # Perform Non Maxima Suppression

            # Display the results after performing NMS
            for (x_tl, y_tl, _, w, h) in detections_fin:
                # Draw the detections
                cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)

            if self.config.VISUALIZE:
                cv2.imshow("Final Detections after applying NMS", clone)

            # print(os.path.split(im_path))
            print(os.path.join(self.config.DIR_PATHS['PRED_SAVE_PH'], os.path.split(im_path)[1]))
            cv2.imwrite(os.path.join(self.config.DIR_PATHS['PRED_SAVE_PH'], os.path.split(im_path)[1]), clone)