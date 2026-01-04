class RuleBasedFeedback:
    def __init__(self):
        pass

    def generate_feedback(self, trick_class, success_prob, features):
        """
        判定結果と特徴量からアドバイスを生成する
        """
        feedback = []
        
        # 成功確率に基づく基本コメント
        if success_prob > 0.8:
            feedback.append("素晴らしい完成度です！")
        elif success_prob > 0.5:
            feedback.append("メイクできていますが、もう少し安定感が出せそうです。")
        else:
            feedback.append("まだメイクが不安定です。")

        # 特徴量に基づく詳細アドバイス（例）
        # features: [shoulder, hip, twist, l_knee, r_knee, center_y, ...]
        
        # 平均的な膝の曲がり具合をチェック
        avg_knee_bend = (features[:, 3].mean() + features[:, 4].mean()) / 2
        if avg_knee_bend > 160: # 180に近いほど伸びている
            feedback.append("膝が伸びきっています。もっと重心を落としましょう。")

        # ねじれの最大値をチェック（回転系の場合）
        max_twist = features[:, 2].max()
        if trick_class == 2: # butter_180
            if max_twist < 30:
                feedback.append("上半身の先行動作（ねじれ）が不足しています。")

        if not feedback:
            feedback.append("フォームは概ね良好です。")

        return "\n".join(feedback)