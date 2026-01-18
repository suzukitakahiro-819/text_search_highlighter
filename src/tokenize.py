"""
トークン化モジュール
単語単位の日本語・英数字トークン分割を実装
"""
import re
from typing import List, Optional


def tokenize_with_fugashi(text: str) -> Optional[List[str]]:
    """
    fugashiを使った単語単位トークン化
    
    Args:
        text: 入力テキスト
        
    Returns:
        トークンリスト（fugashiが利用できない場合はNone）
    """
    try:
        import os
        
        # 設定ファイルパスを検索（インポート前に設定、複数OSに対応）
        # macOS (Homebrew Intel): /usr/local/etc/mecabrc
        # macOS (Homebrew Apple Silicon): /opt/homebrew/etc/mecabrc
        # Linux: /etc/mecabrc
        mecabrc_paths = [
            '/opt/homebrew/etc/mecabrc',  # macOS (Apple Silicon)
            '/usr/local/etc/mecabrc',      # macOS (Intel), Linux (手動インストール)
            '/etc/mecabrc'                 # Linux (標準)
        ]
        
        for path in mecabrc_paths:
            if os.path.exists(path) and 'MECABRC' not in os.environ:
                os.environ['MECABRC'] = path
                break
        
        import fugashi
        
        tagger = fugashi.GenericTagger()
        nodes = tagger(text)
        tokens = [node.surface for node in nodes if node.surface]
        
        return tokens if tokens else None
        
    except (ImportError, Exception):
        # fugashiが利用できない場合（インストールされていない、またはMeCabが設定されていない）
        return None


def tokenize_simple(text: str) -> List[str]:
    """
    簡易トークン化
    - 英数字は単語として分割
    - 日本語は連続する日本語文字列を一塊として扱う
    
    Args:
        text: 入力テキスト
        
    Returns:
        トークンリスト
    """
    if not text:
        return []
    
    # パターン: 英数字の単語、日本語文字列、空白/句読点などを分割
    # \w+ で英数字単語、[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+ で日本語
    # その他の文字（空白、句読点等）は個別に分割
    
    tokens = []
    # 英数字の単語（アルファベットと数字の組み合わせ）
    pattern_word = r'[a-zA-Z0-9]+'
    # 日本語文字（ひらがな、カタカナ、漢字）
    pattern_jp = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+'
    # その他の文字（空白、句読点など）
    pattern_other = r'[^\s]'
    
    # 複数パターンを組み合わせた正規表現
    combined_pattern = rf'({pattern_word}|{pattern_jp}|{pattern_other})'
    
    matches = re.finditer(combined_pattern, text)
    for match in matches:
        token = match.group(0)
        if token.strip():  # 空白のみのトークンは除外
            tokens.append(token)
    
    return tokens


def tokenize(text: str) -> List[str]:
    """
    トークン化のエントリーポイント
    fugashiが利用可能な場合は単語単位でトークン化、
    利用できない場合は簡易版を使用
    
    Args:
        text: 入力テキスト
        
    Returns:
        トークンリスト
    """
    # まずfugashiを試す
    result = tokenize_with_fugashi(text)
    if result is not None:
        return result
    
    # fugashiが使えない場合は簡易版を使用
    return tokenize_simple(text)
