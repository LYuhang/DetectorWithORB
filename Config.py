# -*- coding: utf-8 -*-
import os

'''
This class is used to set params of this project, including 
general config, paths config, feature config.
'''
class Config(object):
    def __init__(self, project_id=None):
        '''
        :param project_id: set project_id
        '''
        ############### General Config ###############
        # define the feature used
        self.DES_TYPE = "ORB"   ## HOG, LBP, ORB
        self.CLF_TYPE = "LIN_SVM"

        # define the project_id
        if project_id:
            self.PROJECT_ID = project_id
        else:
            self.PROJECT_ID = "New_Vedio_New_Neg" + self.DES_TYPE + '_' + self.CLF_TYPE

        # define image processing params
        self.THRESHOLD = 0.3
        self.DOWNSCALE = 1.25

        ################## Pathes ####################
        # define the working dirnames
        self.update_names()
        self.mk_new_dirs()

        ################# K-means ###################
        self.USE_MINIBATCH = True
        self.Kmeans = {
            "Clusters" : 128,
            "batch_size" : 2000
        }

        ################# Features #################
        # Define HOG Features params
        self.MIN_WDW_SIZE = [64, 64]
        self.STEP_SIZE = [12, 12]
        self.ORIENTATIONS = 9
        self.PIXELS_PER_CELL = [3, 3]
        self.CELLS_PER_BLOCK = [3, 3]
        self.VISUALIZE = True
        self.NORMALIZE = True
        self.IF_PRINT = False
        self.KEEP_FEAT = False

        # Define LBP Features params
        self.LBP_RADIUS = 3
        self.LBP_POINTS = 8 * self.LBP_RADIUS

        # Define ORB Features params
        self.ORBParam = {
            "Nfeatures": 200,
            "Edgethresh": 2
        }

    def mk_new_dirs(self):
        for ph in self.DIR_PATHS.values():
            if not os.path.exists(ph):
                os.makedirs(ph)
                print("==> Directory Tree", ph, "created")

    def update_names(self):
        # Pathes
        self.DIR_PATHS = {
            "POS_FEAT_PH": os.path.join("./source/features", self.PROJECT_ID, "pos"),
            "NEG_FEAT_PH": os.path.join("./source/features", self.PROJECT_ID, "neg"),
            "MODEL_DIR_PH": os.path.join("./source/models", self.PROJECT_ID),
            "PRED_SAVE_PH": os.path.join("./source/predictions", self.PROJECT_ID),
            "POS_IMG_PH": "./source/images/pos",
            "NEG_IMG_PH": "./source/images/neg",
            "TEST_IMG_DIR_PH": "./source/test_images"}
        self.MODEL_PH = os.path.join(self.DIR_PATHS["MODEL_DIR_PH"], "svm.model")
        self.TEST_IMG_PH = os.path.join(self.DIR_PATHS["TEST_IMG_DIR_PH"], "test.jpg")
