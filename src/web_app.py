import streamlit as st
import cv2
import tempfile
import os
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
import re
import shutil
import io
import zipfile
from pathlib import Path

# srcディレクトリをパスに追加してモジュールをインポート可能にする
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

from pose.movenet import MoveNet
from preprocessing.extract_features_improved import ImprovedFeatureExtractor
from visualize.draw_pose import PoseVisualizer
from analysis.inference import TrickAnalyzer
from analysis.rules import RuleBasedFeedback
from training.train_model import SnowboardTrainer, load_data
from training.create_labels import LabelManager
from training.label_master import get_label_master

def main():
    st.set_page_config(page_title="Snowboard AI", page_icon="🏂")
    st.title("Snowboard AI Trainer 🏂")
    
    st.sidebar.title("Menu")
    mode = st.sidebar.radio(
        "モード選択", 
        ["解析 (Analyze)", "教師データ追加 (Add Training Data)", "学習 (Train)", "ラベルマスタ管理 (Label Master)", "一括ラベル付け (Batch Labeling)", "データ管理 (Data Management)"]
    )

    if mode == "解析 (Analyze)":
        render_analyze_page()
    elif mode == "教師データ追加 (Add Training Data)":
        render_add_training_data_page()
    elif mode == "ラベルマスタ管理 (Label Master)":
        render_label_master_page()
    elif mode == "一括ラベル付け (Batch Labeling)":
        render_batch_labeling_page()
    elif mode == "データ管理 (Data Management)":
        render_data_management_page()
    else:
        render_train_page()

def render_analyze_page():
    st.header("動画解析")
    uploaded_file = st.file_uploader("スノーボードの動画をアップロードしてください", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # 一時ファイルに保存
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("解析開始"):
            with st.spinner('AIが解析中...'):
                # 特徴量保存パスを生成
                project_root = os.path.dirname(current_dir)
                output_dir = os.path.join(project_root, "data", "output")
                base_filename = os.path.splitext(uploaded_file.name)[0]
                feature_filename = f"{base_filename}_features.npy"
                output_feature_path = os.path.join(output_dir, feature_filename)

                # 解析処理の実行
                result_video_path, trick_name, success_prob, advice = process_video(
                    video_path, 
                    output_feature_path=output_feature_path
                )
            
            st.success("解析完了！")
            
            st.subheader("判定結果")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("技名", trick_name)
            with col2:
                prob_str = f"{success_prob:.1%}" if success_prob else "N/A"
                st.metric("成功確率", prob_str)
            
            st.info(f"**アドバイス:**\n\n{advice}")

            st.subheader("骨格検知結果")
            # 注意: ブラウザによってはエンコード形式の問題で再生できない場合があります
            st.video(result_video_path)
            
            # ダウンロードボタンを追加
            with open(result_video_path, "rb") as f:
                st.download_button(
                    label="解析結果動画をダウンロード",
                    data=f,
                    file_name="analyzed_result.mp4",
                    mime="video/mp4"
                )
            
            # 一時ファイルのクリーンアップ
            try:
                os.remove(result_video_path)
            except:
                pass

def process_video(input_path, output_feature_path=None):
    # 各モジュールの初期化
    pose_estimator = MoveNet()
    feature_extractor = FeatureExtractor()
    visualizer = PoseVisualizer()
    
    # モデルパス（プロジェクトルートからの相対パスを想定）
    model_path = os.path.join(os.path.dirname(current_dir), 'models', 'trained', 'snowboard_lstm.h5')
    analyzer = TrickAnalyzer(model_path)
    feedback_gen = RuleBasedFeedback()

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 出力用一時ファイル
    output_path = os.path.join(tempfile.gettempdir(), 'output_analyzed.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_keypoints = []
    frames_buffer = []

    # プログレスバー
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        keypoints = pose_estimator.run_inference(frame)
        all_keypoints.append(keypoints)
        frames_buffer.append(frame)
        
        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    
    # 特徴量抽出と推論
    all_keypoints_np = np.array(all_keypoints)
    features = feature_extractor.extract(all_keypoints_np)

    # 特徴量を保存
    if output_feature_path:
        try:
            # ディレクトリが存在しない場合は作成
            output_dir = os.path.dirname(output_feature_path)
            os.makedirs(output_dir, exist_ok=True)
            np.save(output_feature_path, features)
            # Streamlitアプリにフィードバック（オプション）
            st.toast(f"特徴量を保存しました: {os.path.basename(output_feature_path)}")
        except Exception as e:
            st.warning(f"特徴量の保存に失敗しました: {e}")
    
    # リサンプリングと推論
    target_timesteps = 60
    current_timesteps = features.shape[0]
    if current_timesteps > 0:
        indices = np.linspace(0, current_timesteps - 1, target_timesteps).astype(int)
        resampled_features = features[indices]
        trick_id, success_prob = analyzer.predict(resampled_features)
    else:
        trick_id, success_prob = None, None

    label_master = get_label_master()
    trick_names = label_master.get_trick_classes_display()
    trick_name = trick_names.get(trick_id, "Unknown") if trick_id is not None else "Unknown"
    
    advice = feedback_gen.generate_feedback(trick_id, success_prob if success_prob else 0, features)

    # 動画書き出し
    for i, frame in enumerate(frames_buffer):
        vis_frame = visualizer.draw(frame, all_keypoints_np[i])
        cv2.putText(vis_frame, f"Trick: {trick_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(vis_frame)
    
    out.release()
    return output_path, trick_name, success_prob, advice

def get_existing_trick_names(features_dir):
    """
    既存の特徴量ファイルから技名のリストを抽出
    
    Returns:
        set: 技名のセット（例：{'nose_ollie', 'nose_ollie720', 'tail_press'}）
    """
    if not os.path.exists(features_dir):
        return set()
    
    trick_names = set()
    pattern = re.compile(r'^(.+)_(\d+)_features\.npy$')
    
    for filename in os.listdir(features_dir):
        if filename.endswith('_features.npy'):
            match = pattern.match(filename)
            if match:
                trick_name = match.group(1)
                trick_names.add(trick_name)
    
    return trick_names

def get_next_sequence_number(features_dir, trick_name):
    """
    指定された技名の次の連番を取得
    
    Args:
        features_dir: 特徴量ディレクトリ
        trick_name: 技名
    
    Returns:
        int: 次の連番（既存ファイルがない場合は1）
    """
    if not os.path.exists(features_dir):
        return 1
    
    pattern = re.compile(rf'^{re.escape(trick_name)}_(\d+)_features\.npy$')
    max_seq = 0
    
    for filename in os.listdir(features_dir):
        match = pattern.match(filename)
        if match:
            seq = int(match.group(1))
            max_seq = max(max_seq, seq)
    
    return max_seq + 1

def render_add_training_data_page():
    st.header("教師データ追加")
    st.write("動画をアップロードして、教師データとして追加できます。")
    
    label_master = get_label_master()
    TRICK_CLASSES = label_master.get_trick_classes_display()
    SUCCESS_LABELS = label_master.get_success_labels_display()
    
    # プロジェクトルートのパス
    project_root = os.path.dirname(current_dir)
    features_dir = os.path.join(project_root, "data", "features")
    labels_file = os.path.join(project_root, "data", "labels", "labels.json")
    
    # ラベルマスタから内部名(name)のリストを取得
    master_internal_names = [v['name'] for v in label_master.get_all_trick_classes_list()]
    # 既存の特徴量ファイルから技名(内部名)を抽出
    existing_feature_names = get_existing_trick_names(features_dir)
    # 両方をマージして一意なリストを作成
    all_internal_names = sorted(list(set(master_internal_names) | existing_feature_names))
    
    # 動画アップロード
    uploaded_file = st.file_uploader("学習用の動画をアップロードしてください", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # ... (中略) ...
        st.write("技名を選択または入力してください。連番は自動で設定されます。")
        
        # 技名選択（既存から選択 or 新規入力）
        trick_name_options = ["新規入力..."] + all_internal_names if all_internal_names else ["新規入力..."]
        selected_option = st.selectbox(
            "技名を選択（または新規入力）",
            options=trick_name_options,
            help="既存の技名から選択するか、新規に入力できます"
        )
        
        if selected_option == "新規入力...":
            trick_name_input = st.text_input(
                "技名を入力（バリエーション含む）",
                value="",
                placeholder="例: nose_ollie または nose_ollie720",
                help="同じ技でもバリエーション（例：nose_ollie720）を区別できます",
                key="new_trick_name"
            )
        else:
            trick_name_input = selected_option
            st.info(f"選択された技名: **{trick_name_input}**")
        
        # 連番を自動計算
        if trick_name_input:
            next_seq = get_next_sequence_number(features_dir, trick_name_input)
            feature_filename = f"{trick_name_input}_{next_seq:03d}_features.npy"
            
            # 既存ファイルの情報を表示
            if existing_trick_names and trick_name_input in existing_feature_names:
                pattern = re.compile(rf'^{re.escape(trick_name_input)}_(\d+)_features\.npy$')
                existing_files = []
                for filename in os.listdir(features_dir):
                    match = pattern.match(filename)
                    if match:
                        existing_files.append(int(match.group(1)))
                
                if existing_files:
                    st.info(f"既存ファイル: {trick_name_input}_{min(existing_files):03d} ～ {trick_name_input}_{max(existing_files):03d}")
            
            st.success(f"生成されるファイル名: `{feature_filename}` (連番: {next_seq:03d})")
        else:
            st.warning("技名を選択または入力してください")
            feature_filename = None
            next_seq = None
        
        # 特徴量抽出ボタン
        if trick_name_input and next_seq is not None and st.button("特徴量を抽出", key="extract_features"):
            with st.spinner('特徴量を抽出中...'):
                try:
                    # 特徴量抽出
                    pose_estimator = MoveNet()
                    feature_extractor = FeatureExtractor()
                    
                    cap = cv2.VideoCapture(video_path)
                    all_keypoints = []
                    
                    progress_bar = st.progress(0)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        keypoints = pose_estimator.run_inference(frame)
                        all_keypoints.append(keypoints)
                        frame_count += 1
                        
                        if total_frames > 0:
                            progress_bar.progress(min(frame_count / total_frames, 1.0))
                    
                    cap.release()
                    
                    if len(all_keypoints) == 0:
                        st.error("動画からフレームを抽出できませんでした。")
                        return
                    
                    # 特徴量抽出
                    all_keypoints_np = np.array(all_keypoints)
                    features = feature_extractor.extract(all_keypoints_np)
                    
                    # ファイル名生成（技名_連番形式）
                    feature_path = os.path.join(features_dir, feature_filename)
                    
                    # 既存ファイルチェック（通常は発生しないはずだが念のため）
                    if os.path.exists(feature_path):
                        st.warning(f"ファイル `{feature_filename}` は既に存在します。上書きしますか？")
                        if not st.button("上書きして保存", key="overwrite"):
                            st.stop()
                    
                    # 保存
                    os.makedirs(features_dir, exist_ok=True)
                    np.save(feature_path, features)
                    
                    st.success(f"特徴量を抽出しました: {feature_filename}")
                    st.info(f"特徴量の形状: {features.shape}")
                    
                    # セッション状態に保存
                    st.session_state['extracted_feature_file'] = feature_filename
                    st.session_state['extracted_features'] = features
                    st.session_state['trick_name'] = trick_name_input
                    st.session_state['sequence_number'] = next_seq
                    
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # ラベル付けセクション
        if 'extracted_feature_file' in st.session_state:
            st.divider()
            st.subheader("ラベル付け")
            
            feature_filename = st.session_state['extracted_feature_file']
            label_manager = LabelManager(labels_file)
            
            # 最後に追加したラベルの成功メッセージを表示
            if 'last_added_label' in st.session_state:
                last_label = st.session_state['last_added_label']
                if last_label['file'] == feature_filename:
                    st.success(f"✅ ラベルを追加しました: {last_label['trick']} ({last_label['success']})")
                    # 一度表示したら削除（次回のリロードでは表示しない）
                    del st.session_state['last_added_label']
            
            # 既存のラベルを確認（複数ラベル対応）
            existing_labels = []
            if feature_filename in label_manager.labels:
                label_data = label_manager.labels[feature_filename]
                if isinstance(label_data, list):
                    existing_labels = label_data
                else:
                    # 後方互換性：単一ラベル形式
                    existing_labels = [label_data]
            
            if existing_labels:
                st.info(f"📋 この特徴量ファイルには既に **{len(existing_labels)} 個**のラベルが付いています:")
                for i, label in enumerate(existing_labels, 1):
                    trick_name = TRICK_CLASSES.get(label['trick_class'], 'Unknown')
                    success_name = SUCCESS_LABELS[label['success_flag']]
                    st.write(f"  {i}. **{trick_name}** ({success_name})")
            
            # 複数ラベル対応：既存のラベル一覧を表示
            st.divider()
            st.write("**➕ 新しいラベルを追加**")
            st.caption("同じ特徴量ファイルに複数のラベルを追加できます。例：nose_ollie（OK）と nose_ollie720（NG）を同時に追加。")
            
            # ラベルマスタからすべての表示名(display_name)を取得
            all_display_names = sorted(list(TRICK_CLASSES.values()))

            # 技名選択UI
            col_trick1, col_trick2 = st.columns([3, 1])
            with col_trick1:
                trick_name_option = st.selectbox(
                    "技名を選択（または新規入力）",
                    options=["新規入力..."] + all_display_names if all_display_names else ["新規入力..."],
                    help="既存の技名から選択するか、新規に入力できます",
                    key="label_trick_name_select"
                )
            
            with col_trick2:
                use_trick_class = st.checkbox("技分類を使用", value=True, help="技分類（0,1,2）を使用する場合はチェック")
            
            display_name_to_id = {v: k for k, v in TRICK_CLASSES.items()}

            # 技分類の決定
            if trick_name_option == "新規入力...":
                trick_name_input_label = st.text_input(
                    "技名を入力",
                    value="",
                    placeholder="例: nose_ollie または nose_ollie720",
                    key="label_trick_name_input"
                )
                if trick_name_input_label:
                    # 新規入力された名前が既存の表示名と一致するかチェック
                    if trick_name_input_label in display_name_to_id:
                        trick_class = display_name_to_id[trick_name_input_label]
                        st.info(f"既存の技として認識: {trick_class} ({trick_name_input_label})")
                    else:
                        trick_class = None # 新しい技なのでIDはまだない

                    if trick_class is None or not use_trick_class:
                        st.warning("新規技名はラベルマスタ管理ページからの追加を推奨します。")
                        trick_class = st.selectbox(
                            "手動で技分類を選択",
                            options=list(TRICK_CLASSES.keys()),
                            format_func=lambda x: f"{x} ({TRICK_CLASSES[x]})",
                            index=0,
                            key="manual_trick_class"
                        )
                    else:
                        st.info(f"推測された技分類: {trick_class} ({TRICK_CLASSES[trick_class]})")
                else:
                    trick_class = st.selectbox(
                        "技分類を選択",
                        options=list(TRICK_CLASSES.keys()),
                        format_func=lambda x: f"{x} ({TRICK_CLASSES[x]})",
                        index=0,
                        key="trick_class_fallback"
                    )
            else:
                # 既存の技名が選択された場合
                selected_trick_name = trick_name_option
                st.info(f"選択された技名: **{selected_trick_name}**")
                
                # 逆引き辞書で技分類IDを直接取得
                trick_class = display_name_to_id.get(selected_trick_name)
                
                if trick_class is None:
                     st.error("選択された技名に対応するIDが見つかりません。")
                     trick_class = 0 # フォールバック
                
                st.info(f"選択された技分類: {trick_class} ({TRICK_CLASSES[trick_class]})")
            
            # 成功判定選択
            selected_success = st.radio(
                "成功判定",
                options=list(SUCCESS_LABELS.keys()),
                format_func=lambda x: f"{x} ({SUCCESS_LABELS[x]})",
                horizontal=True,
                key="success_select"
            )
            success_flag = selected_success
            
            # ラベル保存ボタン
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("ラベルを追加", type="primary"):
                    try:
                        # 複数ラベル対応：同じファイル名に複数のラベルを保存
                        # ラベルファイルの構造を変更：{filename: [label1, label2, ...]}}
                        if feature_filename not in label_manager.labels:
                            label_manager.labels[feature_filename] = []
                        
                        # リスト形式で保存
                        if not isinstance(label_manager.labels[feature_filename], list):
                            # 既存の単一ラベルをリストに変換
                            old_label = label_manager.labels[feature_filename]
                            label_manager.labels[feature_filename] = [old_label]
                        
                        # 新しいラベルを追加
                        new_label = {
                            "trick_class": int(trick_class),
                            "success_flag": int(success_flag)
                        }
                        
                        # 重複チェック
                        if new_label not in label_manager.labels[feature_filename]:
                            label_manager.labels[feature_filename].append(new_label)
                            label_manager.save_labels()
                            
                            st.success(f"✅ ラベルを追加しました！")
                            st.write(f"- 特徴量ファイル: {feature_filename}")
                            st.write(f"- 技分類: {TRICK_CLASSES[trick_class]}")
                            st.write(f"- 成功判定: {SUCCESS_LABELS[success_flag]}")
                            
                            # セッション状態を保持したまま、ページをリロードして更新
                            # これにより、追加したラベルが表示され、次のラベルも追加可能
                            # 注意: st.rerun() の前に st.success などを表示すると、リロード時に消える
                            # そのため、セッション状態に成功メッセージを保存するか、リロード後に表示する
                            st.session_state['last_added_label'] = {
                                'file': feature_filename,
                                'trick': TRICK_CLASSES[trick_class],
                                'success': SUCCESS_LABELS[success_flag]
                            }
                            st.rerun()
                        else:
                            st.warning("⚠️ 同じラベルは既に登録されています。")
                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            with col2:
                if st.button("完了（次へ）"):
                    # セッション状態をクリア
                    if 'extracted_feature_file' in st.session_state:
                        del st.session_state['extracted_feature_file']
                    if 'extracted_features' in st.session_state:
                        del st.session_state['extracted_features']
                    if 'trick_name' in st.session_state:
                        del st.session_state['trick_name']
                    if 'sequence_number' in st.session_state:
                        del st.session_state['sequence_number']
                    st.rerun()
            
            with col3:
                if st.button("キャンセル"):
                    if 'extracted_feature_file' in st.session_state:
                        del st.session_state['extracted_feature_file']
                    if 'extracted_features' in st.session_state:
                        del st.session_state['extracted_features']
                    if 'trick_name' in st.session_state:
                        del st.session_state['trick_name']
                    if 'sequence_number' in st.session_state:
                        del st.session_state['sequence_number']
                    st.rerun()
            
            # 複数ラベル追加の説明
            if existing_labels:
                st.info("💡 **ヒント**: 「ラベルを追加」ボタンを複数回クリックすることで、同じ特徴量ファイルに複数のラベルを追加できます。例：nose_ollie（OK）と nose_ollie720（NG）を同時に追加。")
    
    # 既存のラベル一覧表示と削除機能
    st.divider()
    st.subheader("登録済みラベル一覧・削除")
    
    label_manager = LabelManager(labels_file)
    if label_manager.labels:
        # 統計情報
        total_files = len(label_manager.labels)
        total_labels = sum(
            len(v) if isinstance(v, list) else 1 
            for v in label_manager.labels.values()
        )
        st.info(f"合計 {total_files} ファイル、{total_labels} 件のラベルが登録されています。")
        
        # 削除モード選択
        delete_mode = st.radio(
            "削除モード",
            ["一覧表示", "個別削除", "ファイル単位で削除"],
            horizontal=True,
            key="delete_mode"
        )
        
        if delete_mode == "一覧表示":
            # データフレーム形式で表示（複数ラベル対応）
            labels_data = []
            for feature_file, label_data in label_manager.labels.items():
                # 複数ラベル対応
                if isinstance(label_data, list):
                    for i, label in enumerate(label_data, 1):
                        labels_data.append({
                            "特徴量ファイル": feature_file if i == 1 else "",  # 最初のみ表示
                            "ラベル番号": f"[{i}]" if len(label_data) > 1 else "",
                            "技分類": TRICK_CLASSES.get(label["trick_class"], "Unknown"),
                            "成功判定": SUCCESS_LABELS[label["success_flag"]]
                        })
                else:
                    # 後方互換性：単一ラベル形式
                    labels_data.append({
                        "特徴量ファイル": feature_file,
                        "ラベル番号": "",
                        "技分類": TRICK_CLASSES.get(label_data["trick_class"], "Unknown"),
                        "成功判定": SUCCESS_LABELS[label_data["success_flag"]]
                    })
            
            df = pd.DataFrame(labels_data)
            st.dataframe(df, use_container_width=True)
        
        elif delete_mode == "個別削除":
            st.write("**個別のラベルを削除**")
            
            # 特徴量ファイル選択
            feature_files = sorted(label_manager.labels.keys())
            selected_file = st.selectbox(
                "特徴量ファイルを選択",
                options=feature_files,
                key="delete_file_select"
            )
            
            if selected_file:
                label_data = label_manager.labels[selected_file]
                
                # 複数ラベル対応
                if isinstance(label_data, list):
                    st.write(f"**{selected_file}** のラベル一覧:")
                    for i, label in enumerate(label_data):
                        trick_name = TRICK_CLASSES.get(label["trick_class"], "Unknown")
                        success_name = SUCCESS_LABELS[label["success_flag"]]
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(f"[{i+1}] **{trick_name}** ({success_name})")
                        with col2:
                            if st.button("削除", key=f"delete_label_{selected_file}_{i}", type="secondary"):
                                if label_manager.remove_label(selected_file, label_index=i):
                                    st.success(f"ラベル [{i+1}] を削除しました: {trick_name} ({success_name})")
                                    st.rerun()
                        with col3:
                            st.write("")  # スペーサー
                else:
                    # 単一ラベル形式
                    trick_name = TRICK_CLASSES.get(label_data["trick_class"], "Unknown")
                    success_name = SUCCESS_LABELS[label_data["success_flag"]]
                    st.write(f"**{selected_file}** のラベル:")
                    st.write(f"技分類: {trick_name}, 成功判定: {success_name}")
                    
                    if st.button("このラベルを削除", key=f"delete_single_{selected_file}", type="secondary"):
                        if label_manager.remove_label(selected_file):
                            st.success(f"ラベルを削除しました: {trick_name} ({success_name})")
                            st.rerun()
        
        elif delete_mode == "ファイル単位で削除":
            st.write("**特徴量ファイルのすべてのラベルを削除**")
            st.warning("⚠️ 注意: この操作は、選択したファイルのすべてのラベルを削除します。")
            
            # 特徴量ファイル選択
            feature_files = sorted(label_manager.labels.keys())
            selected_file = st.selectbox(
                "削除する特徴量ファイルを選択",
                options=feature_files,
                key="delete_all_file_select"
            )
            
            if selected_file:
                label_data = label_manager.labels[selected_file]
                
                # ラベル情報を表示
                if isinstance(label_data, list):
                    st.write(f"**{selected_file}** には {len(label_data)} 個のラベルが付いています:")
                    for i, label in enumerate(label_data, 1):
                        trick_name = TRICK_CLASSES.get(label["trick_class"], "Unknown")
                        success_name = SUCCESS_LABELS[label["success_flag"]]
                        st.write(f"  {i}. {trick_name} ({success_name})")
                else:
                    trick_name = TRICK_CLASSES.get(label_data["trick_class"], "Unknown")
                    success_name = SUCCESS_LABELS[label_data["success_flag"]]
                    st.write(f"**{selected_file}** のラベル:")
                    st.write(f"技分類: {trick_name}, 成功判定: {success_name}")
                
                # 確認チェックボックス
                confirm_delete = st.checkbox(
                    f"「{selected_file}」のすべてのラベルを削除することを確認",
                    key="confirm_delete_all"
                )
                
                if confirm_delete:
                    if st.button("すべてのラベルを削除", type="secondary", key="delete_all_button"):
                        if label_manager.remove_label(selected_file):
                            st.success(f"✅ {selected_file} のすべてのラベルを削除しました。")
                            st.rerun()
    else:
        st.info("まだラベルが登録されていません。")

def render_train_page():
    st.header("モデル再学習")
    st.write("データセットを使用してモデルを再学習します。")
    
    project_root = os.path.dirname(current_dir)
    data_dir = st.text_input("データディレクトリ", os.path.join(project_root, "data", "features"))
    labels_file = st.text_input("ラベルファイル", os.path.join(project_root, "data", "labels", "labels.json"))
    model_save_path = st.text_input("モデル保存先", os.path.join(project_root, "models", "trained", "snowboard_lstm.h5"))
    epochs = st.slider("エポック数", 10, 100, 50)
    
    if st.button("学習開始"):
        with st.spinner("学習中..."):
            try:
                X, y_c, y_s = load_data(data_dir, labels_file)
                input_shape = (X.shape[1], X.shape[2])
                
                # ラベルマスタからクラス数を取得
                label_master = get_label_master()
                num_classes = label_master.get_num_classes()
                
                trainer = SnowboardTrainer(input_shape, num_classes=num_classes)
                
                # 学習実行
                history = trainer.train(X, y_c, y_s, epochs=epochs)
                
                # 保存
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                trainer.save_model(model_save_path)
                
                st.success(f"学習完了！モデルを保存しました: {model_save_path}")
                
                # 学習曲線の表示
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Loss (損失)")
                    st.line_chart({
                        'train_loss': history.history['loss'],
                        'val_loss': history.history.get('val_loss', [])
                    })
                
                with col2:
                    st.subheader("Accuracy (精度)")
                    if 'trick_class_accuracy' in history.history:
                        st.line_chart({
                            'train_acc': history.history['trick_class_accuracy'],
                            'val_acc': history.history.get('val_trick_class_accuracy', [])
                        })
                
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                import traceback
                st.code(traceback.format_exc())

def render_label_master_page():
    st.header("ラベルマスタ管理")
    st.write("技の分類（クラス）を編集、追加、削除します。")

    label_master = get_label_master()
    
    # セッション状態に元のデータフレームをキャッシュする
    if 'original_tricks_df' not in st.session_state:
        tricks_list = label_master.get_all_trick_classes_list()
        # idを文字列に変換してからインデックスに設定
        df = pd.DataFrame(tricks_list)
        df['id'] = df['id'].astype(str)
        st.session_state.original_tricks_df = df.set_index('id')

    st.info("テーブルを直接編集、行を追加、または行を選択して削除できます。")
    
    # 編集用データエディタ
    edited_df = st.data_editor(
        st.session_state.original_tricks_df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )

    if st.button("変更を保存", type="primary"):
        original_df = st.session_state.original_tricks_df
        
        # 削除された行を特定
        deleted_ids = set(original_df.index) - set(edited_df.index)
        for trick_id in deleted_ids:
            try:
                label_master.remove_trick_class(int(trick_id))
                st.toast(f"✅ 技 ID: {trick_id} を削除しました。")
            except Exception as e:
                st.error(f"❌ 技 ID: {trick_id} の削除中にエラーが発生しました: {e}")

        # 追加・更新された行を特定
        for trick_id, row in edited_df.iterrows():
            # nameが空、またはNaNの場合はスキップ
            if pd.isna(row['name']) or row['name'].strip() == "":
                continue

            if trick_id not in original_df.index:
                # 新しい行を追加
                try:
                    new_id = label_master.add_trick_class(
                        name=row['name'],
                        display_name=row['display_name'],
                        description=row['description']
                    )
                    st.toast(f"✅ 新しい技 '{row['name']}' (ID: {new_id}) を追加しました。")
                except Exception as e:
                    st.error(f"❌ 新しい技 '{row['name']}' の追加中にエラーが発生しました: {e}")
            else:
                # 既存の行の変更をチェック
                original_row = original_df.loc[trick_id]
                if not original_row.equals(row):
                    try:
                        label_master.update_trick_class(
                            int(trick_id),
                            name=row['name'],
                            display_name=row['display_name'],
                            description=row['description']
                        )
                        st.toast(f"✅ 技 ID: {trick_id} を更新しました。")
                    except Exception as e:
                        st.error(f"❌ 技 ID: {trick_id} の更新中にエラーが発生しました: {e}")
        
        # セッション状態をクリアしてデータを再読み込み
        if 'original_tricks_df' in st.session_state:
            del st.session_state.original_tricks_df
        
        st.success("変更が正常に保存されました。ページを更新します。")
        # 少し待ってからリロード
        import time
        time.sleep(1)
        st.rerun()

def render_batch_labeling_page():
    st.header("一括ラベル付け")
    st.write("`data/output` に保存されている解析済みの特徴量ファイルに一括でラベルを付け、教師データとして登録します。")

    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, "data", "output")
    features_dir = os.path.join(project_root, "data", "features")
    labels_file = os.path.join(project_root, "data", "labels", "labels.json")

    label_manager = LabelManager(labels_file)
    label_master = get_label_master()

    # ラベルマスタから選択肢を取得
    trick_options = list(label_master.get_trick_classes_display().values())
    success_options = list(label_master.get_success_labels_display().values())

    # outputディレクトリの存在チェック
    if not os.path.exists(output_dir) or not any(f.endswith('_features.npy') for f in os.listdir(output_dir)):
        st.info("`data/output` ディレクトリに処理対象のファイルがありません。")
        return

    # ラベル付けされていないnpyファイルを取得
    unlabeled_files = [f for f in os.listdir(output_dir) if f.endswith('_features.npy')]

    if not unlabeled_files:
        st.success("全ての解析済みファイルは既に教師データとして登録されているか、処理対象ファイルが見つかりませんでした。")
        return

    st.subheader("ラベル付け対象ファイル")
    
    # データフレームを作成
    # セッション状態で管理して、リロード後も編集内容を（ある程度）維持する
    if 'batch_labeling_df' not in st.session_state:
        df_data = {
            "ファイル名": unlabeled_files,
            "技名": [None] * len(unlabeled_files),
            "成功判定": [None] * len(unlabeled_files),
            "登録する": [False] * len(unlabeled_files)
        }
        st.session_state.batch_labeling_df = pd.DataFrame(df_data)

    edited_df = st.data_editor(
        st.session_state.batch_labeling_df,
        use_container_width=True,
        column_config={
            "ファイル名": st.column_config.TextColumn(disabled=True),
            "技名": st.column_config.SelectboxColumn("技名", options=trick_options, required=False),
            "成功判定": st.column_config.SelectboxColumn("成功判定", options=success_options, required=False),
            "登録する": st.column_config.CheckboxColumn("登録する", default=False)
        },
        hide_index=True,
        key="batch_labeling_editor"
    )
    # 編集結果をセッションに保存
    st.session_state.batch_labeling_df = edited_df

    if st.button("選択したファイルを一括登録", type="primary"):
        with st.spinner("登録処理中..."):
            # 登録対象の行を取得
            to_register_df = edited_df[edited_df["登録する"] == True]

            if to_register_df.empty:
                st.warning("登録対象のファイルが選択されていません。")
                st.stop()

            # 逆引き辞書を作成
            trick_name_to_id = {v: k for k, v in label_master.get_trick_classes_display().items()}
            success_name_to_id = {v: k for k, v in label_master.get_success_labels_display().items()}
            
            # 内部技名を取得するための辞書
            trick_display_to_internal = {v['display_name']: v['name'] for v in label_master.get_all_trick_classes_list()}

            success_count = 0
            error_count = 0
            processed_files = []

            for _, row in to_register_df.iterrows():
                try:
                    original_filename = row["ファイル名"]
                    display_trick_name = row["技名"]
                    success_name = row["成功判定"]
                    processed_files.append(original_filename)

                    # 必須項目が選択されているかチェック
                    if not display_trick_name or not success_name:
                        st.warning(f"ファイル `{original_filename}` の技名または成功判定が選択されていません。スキップします。")
                        error_count += 1
                        continue

                    # IDと内部技名を取得
                    trick_class_id = trick_name_to_id[display_trick_name]
                    success_flag_id = success_name_to_id[success_name]
                    internal_trick_name = trick_display_to_internal[display_trick_name]

                    # 新しいファイル名を生成
                    next_seq = get_next_sequence_number(features_dir, internal_trick_name)
                    new_filename = f"{internal_trick_name}_{next_seq:03d}_features.npy"
                    
                    # ラベルを追加
                    # add_labelは内部でsave_labelsを呼ぶので、ループ内で何度もファイルI/Oが発生する点に注意
                    label_manager.add_label(
                        feature_file=new_filename,
                        trick_class=trick_class_id,
                        success_flag=success_flag_id
                    )

                    # ファイルを移動＆リネーム
                    original_path = os.path.join(output_dir, original_filename)
                    new_path = os.path.join(features_dir, new_filename)
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    shutil.move(original_path, new_path)
                    
                    st.toast(f"✅ {original_filename} を {new_filename} として登録しました。")
                    success_count += 1

                except Exception as e:
                    st.error(f"❌ {original_filename} の処理中にエラーが発生しました: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    error_count += 1
            
            # セッション状態から処理済みのファイルを削除
            if 'batch_labeling_df' in st.session_state:
                current_df = st.session_state.batch_labeling_df
                st.session_state.batch_labeling_df = current_df[~current_df['ファイル名'].isin(processed_files)]

            st.success(f"処理完了！ {success_count}件のファイルを登録し、{error_count}件のエラーがありました。")
            time.sleep(2) # メッセージ表示のためのウェイト
            st.rerun()

def render_data_management_page():
    """
    データ管理ページを描画する
    """
    st.header("データ管理")
    create_zip_and_download_button()

def create_zip_and_download_button():
    """
    dataディレクトリ全体をzip化し、ダウンロードボタンを表示する
    """
    st.subheader("全データのダウンロード")
    st.write("現在の`data`ディレクトリのすべての内容をzipファイルとしてダウンロードします。")

    project_root = Path(current_dir).parent
    data_dir = project_root / "data"

    if not data_dir.is_dir():
        st.warning("`data`ディレクトリが見つかりません。")
        return

    # メモリ上でzipファイルを作成
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_path in data_dir.glob('**/*'):
            if file_path.is_file():
                # zipファイル内のパスを data/xxx のように設定
                zip_path = file_path.relative_to(data_dir.parent)
                zip_file.write(file_path, arcname=str(zip_path))

    zip_buffer.seek(0)

    st.download_button(
        label="Download `data` directory as .zip",
        data=zip_buffer,
        file_name="snowboard_ai_data.zip",
        mime="application/zip"
    )

if __name__ == "__main__":
    main()
