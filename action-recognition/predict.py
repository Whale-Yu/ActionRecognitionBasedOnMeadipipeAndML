from unittest import mock
import cv2
import numpy as np
import time
import joblib
import poseembedding as pe  # 姿态关键点编码模块
import mediapipe as mp


def predict(model):
    # 导入模型
    pose_knn = joblib.load(model)

    mp_drawing = mp.solutions.drawing_utils  # 画图
    mp_drawing_styles = mp.solutions.drawing_styles  # 渲染风格
    mp_pose = mp.solutions.pose  # 姿势识别
    prevTime = 0
    # 人体姿势关键点位
    keyPoint = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index"
    ]
    # 人体姿势关键坐标（一个点位:(x,y,z)
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
        "left_pinky_y",
        "left_pinky_z",
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
    print(f'keyPoint: {len(keyPoint)}')
    print(f'keyXYZ: {len(keyXYZ)}')

    res_point = []
    poe = pe.FullBodyPoseEmbedder()
    p_landmarks = []

    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            # image = cv2.flip(image, 1)
            # cv2.imwrite('res.png', image)
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                for index, landmarks in enumerate(results.pose_landmarks.landmark):
                    # print(index, landmarks.x, landmarks.y, landmarks.z)
                    res_point.append(landmarks.x)
                    res_point.append(landmarks.y)
                    res_point.append(landmarks.z)
                # print(res_point)
                shape1 = int(len(res_point) / len(keyXYZ))
                res_point = np.array(res_point).reshape(33, 3)
                marks = poe(res_point)
                marks = marks.reshape(shape1, len(keyXYZ))
                pred = pose_knn.predict(marks)
                # print(f'score: {pred}')
                print(pred, pose_knn.predict_proba(marks))
                res_point = []
                colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (118, 25, 124)]
                if pred == 0:
                    cv2.putText(image, "oath", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[0], 2, cv2.LINE_AA)
                elif pred == 1:
                    cv2.putText(image, "put_hand", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[1], 2, cv2.LINE_AA)
                elif pred == 2:
                    cv2.putText(image, "stand", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[2], 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "nothing", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[3], 2, cv2.LINE_AA)

            # 画关键点
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # FPS
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'fps: {int(fps)}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 117, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'exit:esc', (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (245, 120, 117), 2, cv2.LINE_AA)

            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()


if __name__ == '__main__':
    model = 'models/best.joblib'
    predict(model)
