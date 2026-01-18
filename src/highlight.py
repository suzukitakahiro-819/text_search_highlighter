"""
ハイライト生成モジュール
類似度に応じてグラデーションでハイライトしたHTMLを生成
"""
import html
from typing import List, Tuple
import numpy as np


def escape_html(text: str) -> str:
    """HTMLエスケープ"""
    return html.escape(text)


def linear_map(value: float, min_val: float, max_val: float, out_min: float, out_max: float) -> float:
    """
    線形マッピング
    
    Args:
        value: 入力値
        min_val: 入力の最小値
        max_val: 入力の最大値
        out_min: 出力の最小値
        out_max: 出力の最大値
        
    Returns:
        マッピング後の値
    """
    if max_val <= min_val:
        return out_min
    
    # 正規化
    normalized = (value - min_val) / (max_val - min_val)
    # 出力範囲にマッピング
    return out_min + normalized * (out_max - out_min)


def highlight_tokens(
    tokens: List[str],
    scores: np.ndarray,
    threshold: float,
    highlight_type: str = "both"
) -> str:
    """
    トークンリストをハイライトしたHTMLとして生成
    
    Args:
        tokens: トークンリスト
        scores: 類似度スコア配列（shape: [n_tokens]）
        threshold: ハイライト閾値（この値以上をハイライト）
        highlight_type: ハイライトタイプ（"string", "semantic", "both"）
        
    Returns:
        ハイライトされたHTML文字列
    """
    if len(tokens) != len(scores):
        raise ValueError(f"tokens and scores length mismatch: {len(tokens)} != {len(scores)}")
    
    html_parts = []
    
    for token, score in zip(tokens, scores):
        escaped_token = escape_html(token)
        
        if score >= threshold:
            # ハイライト：閾値〜1の範囲で0.2〜0.85に線形マッピング
            alpha = linear_map(score, threshold, 1.0, 0.2, 0.85)
            
            # ハイライトタイプに応じて色を変える
            if highlight_type == "string":
                # 表記類似：黄色系
                color = f"rgba(255, 235, 59, {alpha:.2f})"
            elif highlight_type == "semantic":
                # 意味類似：水色系
                color = f"rgba(64, 224, 208, {alpha:.2f})"
            else:
                # 両方または統合：黄色系
                color = f"rgba(255, 235, 59, {alpha:.2f})"
            
            html_parts.append(f'<span style="background-color: {color}">{escaped_token}</span>')
        else:
            # ハイライトなし
            html_parts.append(escaped_token)
    
    return "".join(html_parts)


def highlight_with_string_and_semantic(
    tokens: List[str],
    string_scores: np.ndarray,
    semantic_scores: np.ndarray,
    string_threshold: float,
    semantic_threshold: float,
    use_string: bool,
    use_semantic: bool,
    show_scores: bool = False,
    is_punctuation_func = None
) -> str:
    """
    文字列類似と意味類似を別々に判定してハイライト
    
    Args:
        tokens: トークンリスト
        string_scores: 文字列類似度スコア（0〜1）
        semantic_scores: 意味類似度スコア（0〜1）
        string_threshold: 文字列類似の閾値（0〜1）
        semantic_threshold: 意味類似の閾値（0〜1）
        use_string: 文字列類似を使うか
        use_semantic: 意味類似を使うか
        show_scores: 各トークンの下に類似度を表示するか
        
    Returns:
        ハイライトされたHTML文字列
    """
    if len(tokens) != len(string_scores) or len(tokens) != len(semantic_scores):
        raise ValueError("tokens and scores length mismatch")
    
    html_parts = []
    
    for i, token in enumerate(tokens):
        escaped_token = escape_html(token)
        
        # 記号かどうかを判定（関数が渡された場合）
        is_punct = False
        if is_punctuation_func is not None:
            is_punct = is_punctuation_func(token)
        
        # 表記類似でスコアが0の場合はハイライトしない
        string_highlight = use_string and string_scores[i] > 0 and string_scores[i] >= string_threshold
        semantic_highlight = use_semantic and semantic_scores[i] >= semantic_threshold
        
        if string_highlight or semantic_highlight:
            # 両方の場合は混色
            if string_highlight and semantic_highlight:
                # 両方の場合は黄色と水色の混色（中間的な色）
                # スコアが高いほど濃くする（0.4〜0.95の範囲）
                string_alpha_val = linear_map(string_scores[i], string_threshold, 1.0, 0.4, 0.95)
                semantic_alpha_val = linear_map(semantic_scores[i], semantic_threshold, 1.0, 0.4, 0.95)
                # 混色：黄色(255,235,59)と水色(64,224,208)を重み付けして混合
                weight_string = string_alpha_val / (string_alpha_val + semantic_alpha_val)
                weight_semantic = semantic_alpha_val / (string_alpha_val + semantic_alpha_val)
                r = int(255 * weight_string + 64 * weight_semantic)
                g = int(235 * weight_string + 224 * weight_semantic)
                b = int(59 * weight_string + 208 * weight_semantic)
                alpha = min(0.95, (string_alpha_val + semantic_alpha_val) / 2)
                color = f"rgba({r}, {g}, {b}, {alpha:.2f})"
            elif string_highlight:
                # 文字列類似のみ：黄色系（スコアが高いほど濃く：0.4〜0.95）
                alpha = linear_map(string_scores[i], string_threshold, 1.0, 0.4, 0.95)
                color = f"rgba(255, 235, 59, {alpha:.2f})"
            else:
                # 意味類似のみ：水色系（スコアが高いほど濃く：0.4〜0.95）
                alpha = linear_map(semantic_scores[i], semantic_threshold, 1.0, 0.4, 0.95)
                color = f"rgba(64, 224, 208, {alpha:.2f})"
            
            if show_scores and not is_punct:
                # 類似度を表示する場合（縦に並べる）（記号は除外）
                score_html_parts = []
                if use_string:
                    score_html_parts.append(f'<span style="font-size: 0.65em; color: #666; line-height: 1.1; display: block;">{string_scores[i]:.3f}</span>')
                if use_semantic:
                    score_html_parts.append(f'<span style="font-size: 0.65em; color: #666; line-height: 1.1; display: block;">{semantic_scores[i]:.3f}</span>')
                score_html = "".join(score_html_parts)
                
                html_parts.append(
                    f'<span style="display: inline-block; vertical-align: top; margin-right: 2px; text-align: center; white-space: nowrap;">'
                    f'<span style="background-color: {color}; padding: 1px 2px; display: inline-block;">{escaped_token}</span>'
                    f'{score_html}'
                    f'</span>'
                )
            else:
                html_parts.append(f'<span style="background-color: {color}">{escaped_token}</span> ')
        else:
            # ハイライトなし
            if show_scores and not is_punct:
                # 類似度を表示する場合（縦に並べる）（記号は除外）
                score_html_parts = []
                if use_string:
                    score_html_parts.append(f'<span style="font-size: 0.65em; color: #999; line-height: 1.1; display: block;">{string_scores[i]:.3f}</span>')
                if use_semantic:
                    score_html_parts.append(f'<span style="font-size: 0.65em; color: #999; line-height: 1.1; display: block;">{semantic_scores[i]:.3f}</span>')
                score_html = "".join(score_html_parts)
                
                html_parts.append(
                    f'<span style="display: inline-block; vertical-align: top; margin-right: 2px; text-align: center; white-space: nowrap;">'
                    f'<span style="padding: 1px 2px; display: inline-block;">{escaped_token}</span>'
                    f'{score_html}'
                    f'</span>'
                )
            else:
                html_parts.append(escaped_token)
    
    return "".join(html_parts)
