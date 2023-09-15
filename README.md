# 👇Action Recognition:Action Recognition Based on Mediapipe and ML(MachineLearning)

目录：

- [👇Action Recognition:Action Recognition Based on Mediapipe and ML(MachineLearning)](#action-recognitionaction-recognition-based-on-mediapipe-and-mlmachinelearning)
    - [🐱Introduce](#introduce)
    - [🐖QuickStart](#quickstart)
        - [Dependencies](#dependencies)
        - [Inference](#inference)
    - [😔Training](#training)
        - [Preprocessing](#preprocessing)
            - [1.提取关键点（详见mediapipe-Fitness-counter-master/code/\&使用说明-yjy.md）](#1提取关键点详见mediapipe-fitness-counter-mastercode使用说明-yjymd)
            - [2.归一化处理（action-recognition）](#2归一化处理action-recognition)
        - [training](#training-1)
            - [3.训练代码（action-recognition）](#3训练代码action-recognition)
        - [predict](#predict)
            - [4.预测代码（action-recognition）](#4预测代码action-recognition)
    - [🐒Reference](#reference)
    - [🐕Thanks](#thanks)

## 🐱Introduce

**Action Recognition Based on Mediapipe and ML(MachineLearning)**

通过利用mediapipe库来提取人体姿态的关键点，共含33个3维的landmarks，用于制作行为或姿态数据集，并采用l多种机器学习算法进行训练，包括如KNN、逻辑回归、决策树和随机森林，所有模型评估的准确率均在90%以上。通过接入摄像头，实现了实时预测功能，可以实现检测人体姿态并进行分类，具有良好的性能，FPS稳定在25左右，并没有明显的延迟问题。

* Demo:

    ![demo.gif](action-recognition/&temp/demo1.webp)

* Accuracy：

| arithmetic                                                                                                               | acc                |
|--------------------------------------------------------------------------------------------------------------------------|--------------------|
| [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)    | 0.9090909090909091 |
| [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)     | 1.0                |
| [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)     | 1.0                |
| [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | 1.0                |

注：以上均在25条训练集、11条测试集上训练、评估所得

## 🐖QuickStart

### Dependencies

``pip install -r requirements.txt
``

### Inference

``python predict.py
``

## 😔Training

### Preprocessing

#### 1.提取关键点（[详见mediapipe-Fitness-counter-master/code/&使用说明-yjy.md](https://github.com/CrabBoss-lab/ActionRecognitionBasedOnMeadipipeAndML/blob/master/mediapipe-Fitness-counter-master/code/%26%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E-yjy.md)）

#### 2.归一化处理（[action-recognition](https://github.com/CrabBoss-lab/ActionRecognitionBasedOnMeadipipeAndML/tree/master/action-recognition)）

* action-recognition下新建fitness_poses_csvs_out文件夹
* 将第一步fitness_poses_csvs_out中的oath.csv复制
* 修改normalize.py中对应csv文件名
* 运行normalize.py
* 修改输出csv文件的第一列(改为对应行为类别的索引0、1、2、3...，如oath为0，put_hand为2，以此类推)
* 以此类推将所有csv处理
* data下为最终处理好的数据集
* 注：action-recognition下新建fitness_poses_csvs_out一次只能处理一个csv

### training

#### 3.训练代码（[action-recognition](https://github.com/CrabBoss-lab/ActionRecognitionBasedOnMeadipipeAndML/tree/master/action-recognition)）

```python train.py```

### predict

#### 4.预测代码（[action-recognition](https://github.com/CrabBoss-lab/ActionRecognitionBasedOnMeadipipeAndML/tree/master/action-recognition)）

```python predict.py```

## 🐒Reference

* [Mediapipe](https://google.github.io/mediapipe/)

* [mediapipe-Fitness-counter](https://github.com/MichistaLin/mediapipe-Fitness-counter)

* [KNN-Fall-Detection](https://github.com/Code-Deer/KNN-Fall-Detection)

## 🐕Thanks

* @Studio:JHC Software Dev Studio

* @Mentor:HuangRiChen

* @Author:YuJunYu、ShenYuXuan
