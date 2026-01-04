import cv2
import numpy as np

class PoseVisualizer:
    def __init__(self):
        # 接続するキーポイントのペア
        self.EDGES = {
            (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
            (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
            (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
            (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
            (12, 14): 'c', (14, 16): 'c'
        }

    def draw(self, frame, keypoints, confidence_threshold=0.3):
        """
        フレームに骨格を描画する
        Args:
            frame: 画像配列
            keypoints: (1, 17, 3)
        """
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        # キーポイントの描画
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

        # ラインの描画
        for edge, color in self.EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        
        return frame