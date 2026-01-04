"""
ラベルマスタ管理モジュール
技分類と成功判定のマスタデータを一元管理する
"""
import os
import json
from pathlib import Path


class LabelMaster:
    """技分類と成功判定のマスタデータを管理するクラス"""
    
    def __init__(self, master_file=None):
        """
        Args:
            master_file: マスタファイルのパス（Noneの場合はデフォルトパスを使用）
        """
        if master_file is None:
            # プロジェクトルートからの相対パス
            current_dir = Path(__file__).resolve().parent
            project_root = current_dir.parent.parent
            master_file = project_root / "data" / "config" / "trick_classes.json"
        
        self.master_file = Path(master_file)
        self.data = self.load_master()
    
    def load_master(self):
        """マスタファイルを読み込む"""
        if self.master_file.exists():
            with open(self.master_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # デフォルトのマスタデータを作成
            default_data = {
                "trick_classes": {
                    "0": {
                        "id": 0,
                        "name": "nose_ollie",
                        "display_name": "Nose Ollie",
                        "description": "ノーズオーリー"
                    },
                    "1": {
                        "id": 1,
                        "name": "tail_press",
                        "display_name": "Tail Press",
                        "description": "テールプレス"
                    },
                    "2": {
                        "id": 2,
                        "name": "butter_180",
                        "display_name": "Butter 180",
                        "description": "バター180"
                    }
                },
                "success_labels": {
                    "0": {
                        "id": 0,
                        "name": "NG",
                        "display_name": "NG（失敗）",
                        "description": "技が未完成または失敗"
                    },
                    "1": {
                        "id": 1,
                        "name": "OK",
                        "display_name": "OK（成功）",
                        "description": "技が成功"
                    }
                },
                "metadata": {
                    "version": "1.0",
                    "last_updated": "2026-01-04",
                    "description": "スノーボードAIの技分類マスタデータ"
                }
            }
            self.save_master(default_data)
            return default_data
    
    def save_master(self, data=None):
        """マスタファイルを保存する"""
        if data is None:
            data = self.data
        
        os.makedirs(self.master_file.parent, exist_ok=True)
        with open(self.master_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_trick_classes(self):
        """技分類の辞書を取得（ID → name）"""
        return {
            int(k): v["name"]
            for k, v in self.data["trick_classes"].items()
        }
    
    def get_trick_classes_display(self):
        """技分類の辞書を取得（ID → display_name）"""
        return {
            int(k): v["display_name"]
            for k, v in self.data["trick_classes"].items()
        }
    
    def get_trick_class_info(self, trick_id):
        """特定の技分類の詳細情報を取得"""
        return self.data["trick_classes"].get(str(trick_id))
    
    def get_success_labels(self):
        """成功判定の辞書を取得（ID → name）"""
        return {
            int(k): v["name"]
            for k, v in self.data["success_labels"].items()
        }
    
    def get_success_labels_display(self):
        """成功判定の辞書を取得（ID → display_name）"""
        return {
            int(k): v["display_name"]
            for k, v in self.data["success_labels"].items()
        }
    
    def add_trick_class(self, name, display_name, description=""):
        """新しい技分類を追加"""
        # 次のIDを計算
        existing_ids = [int(k) for k in self.data["trick_classes"].keys()]
        new_id = max(existing_ids) + 1 if existing_ids else 0
        
        self.data["trick_classes"][str(new_id)] = {
            "id": new_id,
            "name": name,
            "display_name": display_name,
            "description": description
        }
        
        self.save_master()
        return new_id
    
    def update_trick_class(self, trick_id, name=None, display_name=None, description=None):
        """既存の技分類を更新"""
        trick_key = str(trick_id)
        if trick_key not in self.data["trick_classes"]:
            raise ValueError(f"Trick class ID {trick_id} not found")
        
        if name is not None:
            self.data["trick_classes"][trick_key]["name"] = name
        if display_name is not None:
            self.data["trick_classes"][trick_key]["display_name"] = display_name
        if description is not None:
            self.data["trick_classes"][trick_key]["description"] = description
        
        self.save_master()
    
    def remove_trick_class(self, trick_id):
        """技分類を削除"""
        trick_key = str(trick_id)
        if trick_key in self.data["trick_classes"]:
            del self.data["trick_classes"][trick_key]
            self.save_master()
            return True
        return False
    
    def get_num_classes(self):
        """技分類の総数を取得"""
        return len(self.data["trick_classes"])
    
    def get_all_trick_classes_list(self):
        """すべての技分類をリスト形式で取得（UI表示用）"""
        return [
            {
                "id": int(k),
                "name": v["name"],
                "display_name": v["display_name"],
                "description": v.get("description", "")
            }
            for k, v in sorted(self.data["trick_classes"].items(), key=lambda x: int(x[0]))
        ]
    
    def search_trick_by_name(self, name):
        """名前で技分類を検索（部分一致）"""
        results = []
        for k, v in self.data["trick_classes"].items():
            if name.lower() in v["name"].lower() or name.lower() in v["display_name"].lower():
                results.append({
                    "id": int(k),
                    "name": v["name"],
                    "display_name": v["display_name"],
                    "description": v.get("description", "")
                })
        return results


# グローバルインスタンス（シングルトン的な使い方）
_label_master_instance = None

def get_label_master():
    """LabelMasterのグローバルインスタンスを取得"""
    global _label_master_instance
    if _label_master_instance is None:
        _label_master_instance = LabelMaster()
    return _label_master_instance
