import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class MoveNet:
    def __init__(self, model_name="movenet_lightning"):
        """
        MoveNetモデルを初期化する
        """
        if model_name == "movenet_lightning":
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.input_size = 192
        elif model_name == "movenet_thunder":
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.input_size = 256
        else:
            raise ValueError("Unsupported model name")

        self.model = module.signatures['serving_default']

    def run_inference(self, frame):
        """
        フレームから骨格推定を行う
        Args:
            frame: OpenCV画像 (BGR)
        Returns:
            keypoints_with_scores: (1, 17, 3) [y, x, score]
        """
        # 画像の前処理
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), self.input_size, self.input_size)
        input_image = tf.cast(img, dtype=tf.int32)

        # 推論実行
        outputs = self.model(input_image)
        keypoints_with_scores = outputs['output_0'].numpy()
        
        # 形状を (1, 17, 3) に統一する
        keypoints_with_scores = np.reshape(keypoints_with_scores, (1, 17, 3))
        
        return keypoints_with_scores

    def process_video(self, video_path):
        """
        動画ファイル全体を処理するヘルパー（テスト用）
        """
        cap = cv2.VideoCapture(video_path)
        results = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results.append(self.run_inference(frame))
        cap.release()
        return np.array(results)