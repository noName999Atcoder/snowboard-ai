"""
動画管理モジュール
判定時に使用した動画を再学習用に保存・管理する
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
import json


class VideoManager:
    """動画ファイルの保存と管理を行うクラス"""
    
    def __init__(self, video_archive_dir=None):
        """
        Args:
            video_archive_dir: 動画アーカイブディレクトリ（Noneの場合はデフォルトパスを使用）
        """
        if video_archive_dir is None:
            # プロジェクトルートからの相対パス
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            video_archive_dir = project_root / "data" / "analyzed_videos"
        
        self.video_archive_dir = Path(video_archive_dir)
        self.video_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # メタデータファイル
        self.metadata_file = self.video_archive_dir / "metadata.json"
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """メタデータファイルを読み込む"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """メタデータファイルを保存する"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def save_analyzed_video(self, video_path, trick_id=None, success_prob=None, 
                           feature_file=None, original_filename=None):
        """
        解析済み動画を保存する
        
        Args:
            video_path: 保存する動画ファイルのパス
            trick_id: 判定された技のID
            success_prob: 成功確率
            feature_file: 対応する特徴量ファイル名
            original_filename: 元のファイル名
            
        Returns:
            str: 保存されたファイル名
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # タイムスタンプ付きファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 元のファイル名がある場合はそれを使用
        if original_filename:
            base_name = Path(original_filename).stem
            extension = Path(original_filename).suffix or video_path.suffix
        else:
            base_name = video_path.stem
            extension = video_path.suffix
        
        # 保存ファイル名
        saved_filename = f"{timestamp}_{base_name}{extension}"
        saved_path = self.video_archive_dir / saved_filename
        
        # 動画ファイルをコピー
        shutil.copy2(video_path, saved_path)
        
        # メタデータを記録
        self.metadata[saved_filename] = {
            "original_filename": original_filename or video_path.name,
            "saved_at": timestamp,
            "trick_id": trick_id,
            "success_prob": float(success_prob) if success_prob is not None else None,
            "feature_file": feature_file,
            "file_size": saved_path.stat().st_size,
            "used_for_training": False  # 再学習に使用したかどうかのフラグ
        }
        
        self.save_metadata()
        
        print(f"Video saved: {saved_filename}")
        return saved_filename
    
    def get_video_info(self, filename):
        """動画のメタデータを取得"""
        return self.metadata.get(filename)
    
    def list_videos(self, trick_id=None, used_for_training=None):
        """
        保存された動画のリストを取得
        
        Args:
            trick_id: 技IDでフィルタ（Noneの場合は全て）
            used_for_training: 再学習使用フラグでフィルタ（Noneの場合は全て）
            
        Returns:
            list: 動画情報のリスト
        """
        videos = []
        for filename, info in self.metadata.items():
            # フィルタリング
            if trick_id is not None and info.get("trick_id") != trick_id:
                continue
            if used_for_training is not None and info.get("used_for_training") != used_for_training:
                continue
            
            videos.append({
                "filename": filename,
                **info
            })
        
        # タイムスタンプでソート（新しい順）
        videos.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return videos
    
    def mark_as_used_for_training(self, filename):
        """動画を再学習に使用済みとしてマーク"""
        if filename in self.metadata:
            self.metadata[filename]["used_for_training"] = True
            self.save_metadata()
            return True
        return False
    
    def move_to_training(self, filename, input_dir="data/input"):
        """
        保存された動画を再学習用のinputディレクトリにコピー
        
        Args:
            filename: 動画ファイル名
            input_dir: コピー先ディレクトリ
            
        Returns:
            str: コピー先のファイルパス
        """
        video_path = self.video_archive_dir / filename
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {filename}")
        
        # コピー先のディレクトリを作成
        input_dir = Path(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        
        # コピー先のパス
        dest_path = input_dir / filename
        
        # ファイルが既に存在する場合は連番を付ける
        counter = 1
        while dest_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            dest_path = input_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        
        # コピー
        shutil.copy2(video_path, dest_path)
        
        # 再学習使用済みとしてマーク
        self.mark_as_used_for_training(filename)
        
        print(f"Video copied to training data: {dest_path}")
        return str(dest_path)
    
    def delete_video(self, filename):
        """動画を削除"""
        video_path = self.video_archive_dir / filename
        
        if video_path.exists():
            video_path.unlink()
        
        if filename in self.metadata:
            del self.metadata[filename]
            self.save_metadata()
            return True
        
        return False
    
    def get_statistics(self):
        """統計情報を取得"""
        total_videos = len(self.metadata)
        used_for_training = sum(1 for v in self.metadata.values() if v.get("used_for_training", False))
        total_size = sum(v.get("file_size", 0) for v in self.metadata.values())
        
        # 技IDごとの件数
        trick_counts = {}
        for info in self.metadata.values():
            trick_id = info.get("trick_id")
            if trick_id is not None:
                trick_counts[trick_id] = trick_counts.get(trick_id, 0) + 1
        
        return {
            "total_videos": total_videos,
            "used_for_training": used_for_training,
            "not_used": total_videos - used_for_training,
            "total_size_mb": total_size / (1024 * 1024),
            "trick_counts": trick_counts
        }
