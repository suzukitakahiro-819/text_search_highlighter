"""
キャッシュ補助モジュール
テキストが変わらない限りトークン列と埋め込みを再利用
"""
from typing import List, Optional, Tuple
import hashlib


class TokenCache:
    """トークン化結果のキャッシュ"""
    
    def __init__(self):
        self._cache_key: Optional[str] = None
        self._cached_tokens: Optional[List[str]] = None
    
    def get(self, text: str) -> Optional[List[str]]:
        """キャッシュから取得"""
        cache_key = self._compute_key(text)
        if self._cache_key == cache_key and self._cached_tokens is not None:
            return self._cached_tokens
        return None
    
    def set(self, text: str, tokens: List[str]):
        """キャッシュに保存"""
        self._cache_key = self._compute_key(text)
        self._cached_tokens = tokens
    
    def clear(self):
        """キャッシュをクリア"""
        self._cache_key = None
        self._cached_tokens = None
    
    @staticmethod
    def _compute_key(text: str) -> str:
        """テキストのハッシュをキーとして使用"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class EmbeddingCache:
    """埋め込みベクトルのキャッシュ"""
    
    def __init__(self):
        self._cache_key: Optional[str] = None
        self._cached_embeddings: Optional[List[List[float]]] = None
    
    def get(self, text: str) -> Optional[List[List[float]]]:
        """キャッシュから取得"""
        cache_key = self._compute_key(text)
        if self._cache_key == cache_key and self._cached_embeddings is not None:
            return self._cached_embeddings
        return None
    
    def set(self, text: str, embeddings: List[List[float]]):
        """キャッシュに保存"""
        self._cache_key = self._compute_key(text)
        self._cached_embeddings = embeddings
    
    def clear(self):
        """キャッシュをクリア"""
        self._cache_key = None
        self._cached_embeddings = None
    
    @staticmethod
    def _compute_key(text: str) -> str:
        """テキストのハッシュをキーとして使用"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
