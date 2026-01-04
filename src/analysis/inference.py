import tensorflow as tf
import numpy as np

class TrickAnalyzer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load model from {self.model_path}. Inference will fail.")
            print(e)

    def predict(self, features):
        """
        特徴量から技と成功率を予測
        Args:
            features: (time_steps, num_features)
        """
        if self.model is None:
            return None, None

        # バッチ次元追加 (1, time_steps, features)
        input_data = np.expand_dims(features, axis=0)
        
        predictions = self.model.predict(input_data, verbose=0)
        
        # predictions[0] -> class probabilities
        # predictions[1] -> success probability
        class_idx = np.argmax(predictions[0][0])
        success_prob = predictions[1][0][0]
        
        return class_idx, success_prob