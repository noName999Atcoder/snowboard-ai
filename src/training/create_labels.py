"""
教師ラベルを作成・管理するツール
特徴量ファイルと対応するラベル（技分類、成功判定）を管理する
"""
import os
import json
import numpy as np
import argparse
from pathlib import Path
from .label_master import get_label_master


# ラベルマスタから技分類と成功判定を取得
_label_master = get_label_master()
TRICK_CLASSES = _label_master.get_trick_classes()
SUCCESS_LABELS = _label_master.get_success_labels()


class LabelManager:
    def __init__(self, labels_file="data/labels/labels.json"):
        """
        ラベル管理クラス
        
        Args:
            labels_file: ラベルを保存するJSONファイルのパス
        """
        self.labels_file = labels_file
        self.labels = self.load_labels()
    
    def load_labels(self):
        """ラベルファイルを読み込む"""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_labels(self):
        """ラベルファイルを保存する"""
        os.makedirs(os.path.dirname(self.labels_file), exist_ok=True)
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)
    
    def add_label(self, feature_file, trick_class, success_flag, video_file=None):
        """
        ラベルを追加する（複数ラベル対応）
        
        Args:
            feature_file: 特徴量ファイル名（例: "nose_ollie_001_features.npy"）
            trick_class: 技分類 (マスタファイルで定義)
            success_flag: 成功判定 (0: NG, 1: OK)
            video_file: 元動画ファイル名（オプション、再学習用）
        """
        # 複数ラベル対応：リスト形式で保存
        if feature_file not in self.labels:
            self.labels[feature_file] = []
        
        # 既存の単一ラベル形式をリストに変換
        if isinstance(self.labels[feature_file], dict):
            old_label = self.labels[feature_file]
            self.labels[feature_file] = [old_label]
        
        # 新しいラベルを追加（動画ファイル情報も保存）
        new_label = {
            "trick_class": int(trick_class),
            "success_flag": int(success_flag)
        }
        
        # 動画ファイル情報を追加（オプション）
        if video_file:
            new_label["video_file"] = video_file
        
        # 重複チェック
        if new_label not in self.labels[feature_file]:
            self.labels[feature_file].append(new_label)
            self.save_labels()
            print(f"Label added: {feature_file} -> Trick: {TRICK_CLASSES[trick_class]}, Success: {SUCCESS_LABELS[success_flag]}")
        else:
            print(f"Label already exists: {feature_file} -> Trick: {TRICK_CLASSES.get(trick_class, 'Unknown')}, Success: {SUCCESS_LABELS.get(success_flag, 'Unknown')}")
    
    def list_labels(self):
        """登録されているラベルを一覧表示（複数ラベル対応）"""
        if not self.labels:
            print("No labels registered.")
            return
        
        print("\n=== Registered Labels ===")
        for feature_file, label_data in self.labels.items():
            # 複数ラベル対応
            if isinstance(label_data, list):
                for i, label in enumerate(label_data, 1):
                    trick_name = TRICK_CLASSES.get(label["trick_class"], "Unknown")
                    success_name = SUCCESS_LABELS.get(label["success_flag"], "Unknown")
                    video_info = f" (video: {label['video_file']})" if label.get("video_file") else ""
                    if len(label_data) > 1:
                        print(f"{feature_file} [{i}]: {trick_name} ({success_name}){video_info}")
                    else:
                        print(f"{feature_file}: {trick_name} ({success_name}){video_info}")
            else:
                # 後方互換性：単一ラベル形式
                trick_name = TRICK_CLASSES.get(label_data["trick_class"], "Unknown")
                success_name = SUCCESS_LABELS.get(label_data["success_flag"], "Unknown")
                video_info = f" (video: {label_data['video_file']})" if label_data.get("video_file") else ""
                print(f"{feature_file}: {trick_name} ({success_name}){video_info}")
    
    def remove_label(self, feature_file, label_index=None):
        """
        ラベルを削除する（複数ラベル対応）
        
        Args:
            feature_file: 特徴量ファイル名
            label_index: 削除するラベルのインデックス（Noneの場合はすべてのラベルを削除）
        """
        if feature_file not in self.labels:
            print(f"Label not found: {feature_file}")
            return False
        
        label_data = self.labels[feature_file]
        
        if label_index is None:
            # すべてのラベルを削除
            del self.labels[feature_file]
            self.save_labels()
            print(f"All labels removed: {feature_file}")
            return True
        else:
            # 複数ラベル対応：特定のラベルのみ削除
            if isinstance(label_data, list):
                if 0 <= label_index < len(label_data):
                    removed_label = label_data.pop(label_index)
                    # ラベルが0個になったらファイル自体を削除
                    if len(label_data) == 0:
                        del self.labels[feature_file]
                    else:
                        self.save_labels()
                    trick_name = TRICK_CLASSES.get(removed_label["trick_class"], "Unknown")
                    success_name = SUCCESS_LABELS[removed_label["success_flag"]]
                    self.save_labels()
                    print(f"Label removed: {feature_file} [{label_index+1}] -> {trick_name} ({success_name})")
                    return True
                else:
                    print(f"Invalid label index: {label_index}")
                    return False
            else:
                # 単一ラベル形式：ファイル全体を削除
                del self.labels[feature_file]
                self.save_labels()
                print(f"Label removed: {feature_file}")
                return True


def interactive_labeling(features_dir="data/features", labels_file="data/labels/labels.json"):
    """
    対話的にラベル付けを行う
    
    Args:
        features_dir: 特徴量ファイルが保存されているディレクトリ
        labels_file: ラベルファイルのパス
    """
    manager = LabelManager(labels_file)
    label_master = get_label_master()
    
    # 特徴量ファイル一覧取得
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('_features.npy')]
    
    if not feature_files:
        print(f"No feature files found in {features_dir}")
        return
    
    print("\n=== Feature Files ===")
    for i, f in enumerate(feature_files, 1):
        status = "✓" if f in manager.labels else " "
        print(f"{i}. [{status}] {f}")
    
    print("\n=== Trick Classes ===")
    trick_classes_display = label_master.get_trick_classes_display()
    for k, v in sorted(trick_classes_display.items()):
        print(f"  {k}: {v}")
    
    print("\n=== Success Flags ===")
    success_labels_display = label_master.get_success_labels_display()
    for k, v in sorted(success_labels_display.items()):
        print(f"  {k}: {v}")
    
    print("\nEnter feature file number to label (or 'q' to quit):")
    while True:
        choice = input("> ").strip()
        
        if choice.lower() == 'q':
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(feature_files):
                feature_file = feature_files[idx]
                
                # 既存のラベルを表示
                if feature_file in manager.labels:
                    existing = manager.labels[feature_file]
                    print(f"\nCurrent label: Trick={TRICK_CLASSES[existing['trick_class']]}, Success={SUCCESS_LABELS[existing['success_flag']]}")
                    update = input("Update? (y/n): ").strip().lower()
                    if update != 'y':
                        continue
                
                # 技分類入力
                max_trick_id = max(label_master.get_trick_classes().keys())
                trick_input = input(f"Trick class (0-{max_trick_id}): ").strip()
                try:
                    trick_class = int(trick_input)
                    if trick_class not in label_master.get_trick_classes():
                        print("Invalid trick class")
                        continue
                except ValueError:
                    print("Invalid input")
                    continue
                
                # 成功判定入力
                success_input = input(f"Success flag (0: NG, 1: OK): ").strip()
                try:
                    success_flag = int(success_input)
                    if success_flag not in label_master.get_success_labels():
                        print("Invalid success flag")
                        continue
                except ValueError:
                    print("Invalid input")
                    continue
                
                # ラベル追加
                manager.add_label(feature_file, trick_class, success_flag)
                print()
            else:
                print("Invalid number")
        except ValueError:
            print("Invalid input")


def main():
    parser = argparse.ArgumentParser(description='Manage training labels')
    parser.add_argument('--mode', type=str, choices=['add', 'list', 'remove', 'interactive'],
                       default='interactive', help='Operation mode')
    parser.add_argument('--feature-file', type=str, help='Feature file name')
    parser.add_argument('--trick-class', type=int, help='Trick class (0-2)')
    parser.add_argument('--success-flag', type=int, help='Success flag (0: NG, 1: OK)')
    parser.add_argument('--labels-file', type=str, default='data/labels/labels.json',
                       help='Labels file path')
    parser.add_argument('--features-dir', type=str, default='data/features',
                       help='Features directory')
    
    args = parser.parse_args()
    
    manager = LabelManager(args.labels_file)
    
    if args.mode == 'interactive':
        interactive_labeling(args.features_dir, args.labels_file)
    elif args.mode == 'add':
        if not all([args.feature_file, args.trick_class is not None, args.success_flag is not None]):
            print("Error: --feature-file, --trick-class, and --success-flag are required for 'add' mode")
            return
        manager.add_label(args.feature_file, args.trick_class, args.success_flag)
    elif args.mode == 'list':
        manager.list_labels()
    elif args.mode == 'remove':
        if not args.feature_file:
            print("Error: --feature-file is required for 'remove' mode")
            return
        manager.remove_label(args.feature_file)


if __name__ == "__main__":
    main()
