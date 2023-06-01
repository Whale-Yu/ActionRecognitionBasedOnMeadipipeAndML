import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import extracttrainingsetkeypoints as ek  # 提取训练集关键点特征
import poseembedding as pe  # 姿态关键点编码模块
import os
import csv

keyXYZ = [
    "nose_x",
    "nose_y",
    "nose_z",
    "left_eye_inner_x",
    "left_eye_inner_y",
    "left_eye_inner_z",
    "left_eye_x",
    "left_eye_y",
    "left_eye_z",
    "left_eye_outer_x",
    "left_eye_outer_y",
    "left_eye_outer_z",
    "right_eye_inner_x",
    "right_eye_inner_y",
    "right_eye_inner_z",
    "right_eye_x",
    "right_eye_y",
    "right_eye_z",
    "right_eye_outer_x",
    "right_eye_outer_y",
    "right_eye_outer_z",
    "left_ear_x",
    "left_ear_y",
    "left_ear_z",
    "right_ear_x",
    "right_ear_y",
    "right_ear_z",
    "mouth_left_x",
    "mouth_left_y",
    "mouth_left_z",
    "mouth_right_x",
    "mouth_right_y",
    "mouth_right_z",
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_z",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_z",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "left_pinky_x",
    "left_pinky_z",
    "left_pinky_y",
    "right_pinky_x",
    "right_pinky_y",
    "right_pinky_z",
    "left_index_x",
    "left_index_y",
    "left_index_z",
    "right_index_x",
    "right_index_y",
    "right_index_z",
    "left_thumb_x",
    "left_thumb_y",
    "left_thumb_z",
    "right_thumb_x",
    "right_thumb_y",
    "right_thumb_z",
    "left_hip_x",
    "left_hip_y",
    "left_hip_z",
    "right_hip_x",
    "right_hip_y",
    "right_hip_z",
    "left_knee_x",
    "left_knee_y",
    "left_knee_z",
    "right_knee_x",
    "right_knee_y",
    "right_knee_z",
    "left_ankle_x",
    "left_ankle_y",
    "left_ankle_z",
    "right_ankle_x",
    "right_ankle_y",
    "right_ankle_z",
    "left_heel_x",
    "left_heel_y",
    "left_heel_z",
    "right_heel_x",
    "right_heel_y",
    "right_heel_z",
    "left_foot_index_x",
    "left_foot_index_y",
    "left_foot_index_z",
    "right_foot_index_x",
    "right_foot_index_y",
    "right_foot_index_z"
]
# print(len(keyXYZ))

# 坐标点预处理
pose_samples_folder = 'fitness_poses_csvs_out'
file_extension = 'csv'
file_separator = ','
n_landmarks = 33
n_dimensions = 3
file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]
poe = pe.FullBodyPoseEmbedder()
p_landmarks = []
for file_name in file_names:
    # 使用文件名作为姿势类名称.
    class_name = file_name[:-(len(file_extension) + 1)]

    # Parse CSV.
    with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
            assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
            landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
            print(landmarks)
            p_landmarks.append(poe(landmarks))

print(p_landmarks)
print(len(p_landmarks))
shape1 = len(p_landmarks)
p_landmarks = np.array(p_landmarks).reshape(shape1, len(keyXYZ))

data = pd.DataFrame(data=p_landmarks, columns=keyXYZ, index=None)
# print(data.iloc[:,0])
data.to_csv("stand.csv", encoding='utf-8')
