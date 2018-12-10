# README
_ _ _

**Abstract**：这个部分用于使用传统图像特征的方法，基于滑动窗口的的目标检测
模块，包含Config模块，FeatureExtractor模块，和Classifier模块，下面将简单
介绍使用方法和各模块的功能。
### Acknowledge
+ Code framework from [Zhongyang Zhang](https://github.com/miracleyoo/opencv-object-detector)

### 使用方法
#### 训练
```commandline
python main.py --action train
```
#### 测试
```commandline
python main.py --action test
```

### 各模块功能
| Config | FeatureExtractor | Classifier |
|:-------:|:---------------:|:----------:|
|定义project的各种参量，包括使用的feature类型，project_id，路径（训练集/
测试集/模型）以及各种feature提取的参数| 定义特征提取函数，该模块包含通用的
图像处理函数，以及特定的feature提取函数| 定义分类器的训练以及预测函数 |

#### Config
| Param | Definition |
|:-----:|:-----------:|
| DES_TYPE | The feature extraction type |
| CLF_TYPE | The classifier model(eg.SVM,MLP) |
| project_id | Self_defined project name |
| THRESHOLD | The threshold when applying nms() |
| DOWNSCALE | The downscale when applying pyramid_gaussian() |
| MIN_WDW_SIZE | The min size of windows detected |
| STEP_SIZE | The slide step when sliding across the image|

#### FeatureExtractor
| Function | Definition |
|:--------:|:----------:|
| resize_crop_by_short() | Resize and crop the input image, output image shape(short_len, short_len)  |
| resize_by_short() | Resize the image |
| image_preprocess_size() | Resize and crop all training images|
| sliding_window() | Slide window at a fix window size |
| overlapping_area() | Calculate the overlap area of two detections |
| nms() | Apply NMS|
| process_image() | Extract features from input image |
| extract_features() | Extract features of all input images |

#### Classifier
| Function | Definition |
|:--------:|:---------:|
|load_data() | Load all features of images|
|train_classifier() | Train the model |
|load_model() | Load the model |
| predict() | Predict the class or score of the input image |
| test_classifier() | Test classifier on the test images |