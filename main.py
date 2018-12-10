# -*- coding: utf-8 -*-

from Config import Config
from Classifier import Classifier
from FeatureExtractor import Extractor
import warnings
import argparse
warnings.filterwarnings("ignore")

## Add a args parser
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--project_id', type=str, help='The name of the project')
parser.add_argument('-a', "--action", type=str, help='Train, test or predict',
                    default="train", choices=["train", "test"])
args = parser.parse_args()

if __name__ == "__main__":
    ## Initialize the config
    conf = Config()

    if args.project_id:
        conf.PROJECT_ID = args.project_id

    if args.action == "train":
        ## Initialize the Extractor and generate training data
        extractor = Extractor(conf)
        extractor.image_preprocess_size()
        extractor.extract_features()

        ## Initialize the Classifier and train the model
        classifier = Classifier(conf)
        classifier.load_data()
        classifier.train_classifier()
    elif args.action == "test":
        ## Initialize the Classifier , load the model and test
        ## on the test images
        classifier = Classifier(conf)
        classifier.load_model()
        classifier.test_classifier()
    else:
        raise Exception("There is no action %s" % args.action)