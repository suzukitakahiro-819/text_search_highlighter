"""
類似度計算モジュール
文字列類似（rapidfuzz）と意味類似（sentence-transformers）を実装
"""
from typing import List, Optional
import numpy as np
from rapidfuzz import fuzz


class StringSimilarity:
    """文字列類似度計算（OCR誤り・表記ブレに強い）"""
    
    @staticmethod
    def similarity(query: str, token: str) -> float:
        """
        文字列類似度を計算（0〜100）
        OCR誤りや表記ブレを考慮して複数のメトリクスを組み合わせ
        
        Args:
            query: 検索クエリ
            token: トークン
            
        Returns:
            類似度スコア（0〜100）
        """
        # 基本類似度
        ratio = fuzz.ratio(query, token)
        
        # 部分マッチ（OCR誤字に強い）
        partial_ratio = fuzz.partial_ratio(query, token)
        
        # トークン順序を無視した類似度
        token_sort_ratio = fuzz.token_sort_ratio(query, token)
        
        # トークンセット類似度
        token_set_ratio = fuzz.token_set_ratio(query, token)
        
        # 複数のメトリクスの最大値を取る（最も寛容な類似度）
        # これにより、OCR誤りや表記ブレに対応
        max_similarity = max(ratio, partial_ratio, token_sort_ratio, token_set_ratio)
        
        # クエリがトークンに含まれている場合は加点
        if query in token or token in query:
            max_similarity = max(max_similarity, 85.0)
        
        return max_similarity


class SemanticSimilarity:
    """意味類似度計算（埋め込みベクトル）"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        """
        初期化
        
        Args:
            model_name: sentence-transformersのモデル名
        """
        self.model_name = model_name
        self._model = None
    
    def _get_model(self):
        """モデルを遅延読み込み"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Please install it with: pip install sentence-transformers"
                )
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        テキストを埋め込みベクトルに変換
        
        Args:
            texts: テキストリスト
            
        Returns:
            埋め込みベクトル配列（shape: [n_texts, embedding_dim]）
        """
        if not texts:
            return np.array([])
        
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def similarity(self, query_embedding: np.ndarray, token_embeddings: np.ndarray) -> np.ndarray:
        """
        コサイン類似度を計算（0〜1）
        
        Args:
            query_embedding: クエリの埋め込みベクトル（shape: [embedding_dim]）
            token_embeddings: トークンの埋め込みベクトル（shape: [n_tokens, embedding_dim]）
            
        Returns:
            類似度スコア配列（shape: [n_tokens]）
        """
        # 正規化
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        token_norms = token_embeddings / (np.linalg.norm(token_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # コサイン類似度
        similarities = np.dot(token_norms, query_norm)
        
        # 0〜1に正規化（コサイン類似度は-1〜1だが、通常は0〜1の範囲で扱う）
        similarities = (similarities + 1.0) / 2.0
        
        return similarities


def combine_scores(
    string_scores: np.ndarray,
    semantic_scores: np.ndarray,
    use_string: bool,
    use_semantic: bool,
    string_weight: float = 0.5
) -> np.ndarray:
    """
    文字列類似度と意味類似度を統合
    
    Args:
        string_scores: 文字列類似度スコア（0〜100）
        semantic_scores: 意味類似度スコア（0〜1）
        use_string: 文字列類似を使うか
        use_semantic: 意味類似を使うか
        string_weight: 文字列類似の重み（0〜1、意味類似の重みは1-string_weight）
        
    Returns:
        統合スコア（0〜1）
    """
    if not use_string and not use_semantic:
        return np.zeros_like(string_scores)
    
    # 文字列類似度を0〜1に正規化
    normalized_string = string_scores / 100.0 if use_string else np.zeros_like(string_scores)
    
    # 意味類似度（既に0〜1）
    normalized_semantic = semantic_scores if use_semantic else np.zeros_like(semantic_scores)
    
    # 両方使う場合は重み付き平均
    if use_string and use_semantic:
        combined = string_weight * normalized_string + (1 - string_weight) * normalized_semantic
    elif use_string:
        combined = normalized_string
    else:
        combined = normalized_semantic
    
    return combined
