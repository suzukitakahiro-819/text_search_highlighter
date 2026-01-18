"""
OCRテキストの単語検索＋類似語/類似表記をグラデーションでハイライトするStreamlitアプリ
"""
import streamlit as st
import numpy as np
import time
import pandas as pd
from typing import List, Tuple

from src.tokenize import tokenize
from src.cache import TokenCache, EmbeddingCache
from src.similarity import StringSimilarity, SemanticSimilarity, combine_scores
from src.highlight import highlight_with_string_and_semantic


# デフォルトの日本語句読点と一般的な記号（グローバルで定義）
DEFAULT_PUNCTUATION_CHARS = set('、。，．.,;:!?・（）()[]【】「」『』〈〉《》""''「」『』【】〔〕〖〗〘〙〚〛‵′″‴‶‷❝❞❛❜❮❯⟦⟧⟨⟩⟪⟫｟｠⸨⸩「」『』【】〔〕〖〗〘〙〚〛〈〉《》『』「」【】')

def is_punctuation(token: str, custom_punctuation: set = None) -> bool:
    """
    記号かどうかを判定（デフォルト記号のみ）
    
    Args:
        token: トークン文字列
        custom_punctuation: 互換性のためのパラメータ（使用しない）
        
    Returns:
        記号の場合True、それ以外False
    """
    # 記号のみで構成されているかチェック（空白は除外）
    token_stripped = token.strip()
    if not token_stripped:
        return False
    
    # すべての文字がデフォルト記号文字セットに含まれているかチェック
    return all(c in DEFAULT_PUNCTUATION_CHARS for c in token_stripped)


def is_excluded_word(token: str, excluded_words: set = None) -> bool:
    """
    除外対象の単語かどうかを判定（デフォルト記号 + 除外単語リスト）
    
    Args:
        token: トークン文字列
        excluded_words: 除外単語のセット（オプション）
        
    Returns:
        除外対象の場合True、それ以外False
    """
    # デフォルト記号かチェック
    if is_punctuation(token):
        return True
    
    # 除外単語リストに含まれているかチェック
    if excluded_words is not None and token in excluded_words:
        return True
    
    return False


def get_unique_word_scores(tokens, scores, excluded_words):
    """
    ユニークな単語ごとの最高スコアを取得
    
    Args:
        tokens: トークンリスト
        scores: スコア配列（numpy配列またはリスト）
        excluded_words: 除外単語のセット
        
    Returns:
        単語をキー、最高スコアを値とする辞書
    """
    token_score_dict = {}
    for i, token in enumerate(tokens):
        # 除外単語は除外
        if not is_excluded_word(token, excluded_words):
            if token not in token_score_dict:
                token_score_dict[token] = scores[i]
            else:
                token_score_dict[token] = max(token_score_dict[token], scores[i])
    return token_score_dict


# ページ設定
st.set_page_config(
    page_title="OCR Text Search Highlighter",
    layout="wide"
)

st.title("OCRテキスト検索ハイライター")
st.markdown("検索語を入力すると、表記類似と意味類似でトークンをハイライト表示します。")


# セッション状態の初期化
if 'token_cache' not in st.session_state:
    st.session_state.token_cache = TokenCache()
if 'embedding_cache' not in st.session_state:
    st.session_state.embedding_cache = EmbeddingCache()
if 'semantic_similarity' not in st.session_state:
    st.session_state.semantic_similarity = SemanticSimilarity()
if 'string_scores' not in st.session_state:
    st.session_state.string_scores = None
if 'semantic_scores' not in st.session_state:
    st.session_state.semantic_scores = None
if 'last_text_input' not in st.session_state:
    st.session_state.last_text_input = None
if 'last_query_input' not in st.session_state:
    st.session_state.last_query_input = None
if 'last_tokens' not in st.session_state:
    st.session_state.last_tokens = None
if 'custom_excluded_words' not in st.session_state:
    st.session_state.custom_excluded_words = set()


# サイドバー：設定
st.sidebar.header("検索設定")

use_string = st.sidebar.checkbox("表記類似を使う", value=True)
use_semantic = st.sidebar.checkbox("意味類似を使う", value=True)

# 閾値入力方法の選択
threshold_input_method = st.sidebar.radio(
    "閾値入力方法",
    ["自動設定", "手動設定"],
    index=0
)

# 初期値（後で更新される可能性がある）
default_string_threshold = 0.3  # 0〜1スケール（旧30を0.3に変更）
default_semantic_threshold = 0.75

if threshold_input_method == "自動設定":
    auto_highlight_count = st.sidebar.number_input(
        "上位何単語をハイライトするか",
        min_value=1,
        max_value=1000,
        value=10,
        step=1,
        help="スコアが高い順にこの数のユニークな単語がハイライトされる閾値を自動計算します（同じ単語の複数出現は1回としてカウント）"
    )
    
    # 過去のスコアから閾値を計算（スコアが利用可能な場合）
    if st.session_state.string_scores is not None and use_string and st.session_state.last_tokens is not None:
        # ユニークな単語ごとの最高スコアを取得
        token_score_dict = get_unique_word_scores(
            st.session_state.last_tokens,
            np.array(st.session_state.string_scores),
            st.session_state.custom_excluded_words
        )
        if token_score_dict:
            sorted_string_scores = sorted(token_score_dict.values(), reverse=True)
            if len(sorted_string_scores) >= auto_highlight_count:
                default_string_threshold = float(sorted_string_scores[auto_highlight_count - 1])
            else:
                default_string_threshold = float(sorted_string_scores[-1]) if len(sorted_string_scores) > 0 else 0.3
    
    if st.session_state.semantic_scores is not None and use_semantic and st.session_state.last_tokens is not None:
        # ユニークな単語ごとの最高スコアを取得
        token_score_dict = get_unique_word_scores(
            st.session_state.last_tokens,
            np.array(st.session_state.semantic_scores),
            st.session_state.custom_excluded_words
        )
        if token_score_dict:
            sorted_semantic_scores = sorted(token_score_dict.values(), reverse=True)
            if len(sorted_semantic_scores) >= auto_highlight_count:
                default_semantic_threshold = float(sorted_semantic_scores[auto_highlight_count - 1])
            else:
                default_semantic_threshold = float(sorted_semantic_scores[-1]) if len(sorted_semantic_scores) > 0 else 0.75

if threshold_input_method == "手動設定":
    # 手動設定内での入力方法を選択
    manual_input_method = st.sidebar.radio(
        "入力方法",
        ["スライダー", "手入力"],
        index=0,
        horizontal=True
    )
    
    if manual_input_method == "スライダー":
        # スライダーの範囲を計算されたスコアの最小〜最大に設定
        if st.session_state.string_scores is not None and use_string:
            string_min = float(np.min(st.session_state.string_scores))
            string_max = float(np.max(st.session_state.string_scores))
            string_default = float(default_string_threshold) if default_string_threshold >= string_min and default_string_threshold <= string_max else string_min
        else:
            string_min = 0.0
            string_max = 1.0
            string_default = float(default_string_threshold)
        
        if st.session_state.semantic_scores is not None and use_semantic:
            semantic_min = float(np.min(st.session_state.semantic_scores))
            semantic_max = float(np.max(st.session_state.semantic_scores))
            semantic_default = float(default_semantic_threshold) if default_semantic_threshold >= semantic_min and default_semantic_threshold <= semantic_max else semantic_min
        else:
            semantic_min = 0.0
            semantic_max = 1.0
            semantic_default = float(default_semantic_threshold)
        
        # スライダーのstepを適切に設定
        string_range = string_max - string_min
        if string_range > 0:
            string_step = max(0.01, string_range / 100)
        else:
            string_step = 0.01
        
        semantic_range = semantic_max - semantic_min
        if semantic_range > 0:
            semantic_step = max(0.001, semantic_range / 100)
        else:
            semantic_step = 0.01
        
        string_threshold = st.sidebar.slider(
            "表記類似しきい値",
            min_value=float(string_min),
            max_value=float(string_max),
            value=float(string_default),
            step=float(string_step)
        )
        
        semantic_threshold = st.sidebar.slider(
            "意味類似しきい値",
            min_value=float(semantic_min),
            max_value=float(semantic_max),
            value=float(semantic_default),
            step=float(semantic_step)
        )
    elif manual_input_method == "手入力":
        string_threshold = st.sidebar.number_input(
            "表記類似しきい値",
            min_value=0.0,
            max_value=1.0,
            value=float(default_string_threshold),
            step=0.01,
            format="%.3f"
        )
        
        semantic_threshold = st.sidebar.number_input(
            "意味類似しきい値",
            min_value=0.0,
            max_value=1.0,
            value=float(default_semantic_threshold),
            step=0.01,
            format="%.3f"
        )
else:  # 自動設定
    string_threshold = default_string_threshold
    semantic_threshold = default_semantic_threshold
    
    st.sidebar.markdown(
        f'<div class="stInfo" style="padding: 0.5rem 1rem; border-radius: 0.25rem; background-color: rgb(240, 253, 244); border-left: 0.25rem solid rgb(34, 197, 94);">'
        f'<div>表記類似閾値: {string_threshold:.3f}</div>'
        f'<div>意味類似閾値: {semantic_threshold:.3f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("")  # 隙間を空ける

# 記号管理
with st.sidebar.expander("除外する単語リスト", expanded=False):
    st.markdown("以下の単語は類似度計算から除外されます。")
    
    # デフォルト記号の表示
    sorted_default_punctuation = sorted(DEFAULT_PUNCTUATION_CHARS)
    st.write(f"**デフォルト記号（{len(sorted_default_punctuation)}個）:**")
    cols_per_row = 5
    for idx in range(0, len(sorted_default_punctuation), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            if idx + col_idx < len(sorted_default_punctuation):
                punct_char = sorted_default_punctuation[idx + col_idx]
                with col:
                    st.text(f" {punct_char}")
    
    # 現在の除外単語を表示と削除
    if st.session_state.custom_excluded_words:
        sorted_words = sorted(st.session_state.custom_excluded_words)
        st.write(f"**除外単語（{len(sorted_words)}個）:**")
        
        # 単語を列挙して削除ボタンを表示
        for word in sorted_words:
            col1, col2 = st.columns([1, 10])
            with col1:
                if st.button("×", key=f"remove_word_{word}", help=f"「{word}」を削除"):
                    st.session_state.custom_excluded_words.discard(word)
                    st.success(f"「{word}」を削除しました")
                    st.rerun()
            with col2:
                st.text(word)
    
    # 単語の追加
    new_word = st.text_input(
        "単語を追加",
        value="",
        help="除外したい単語を入力してください（例: の、は、です）。",
        key="new_word_input"
    )
    if st.button("単語を追加"):
        if new_word and new_word.strip():
            word = new_word.strip()
            if word not in st.session_state.custom_excluded_words:
                st.session_state.custom_excluded_words.add(word)
                st.success(f"「{word}」を追加しました")
                st.rerun()
            else:
                st.warning(f"「{word}」は既に追加されています。")
        else:
            st.warning("単語を入力してください。")
    
    # 除外単語のリセット
    if st.button("すべて削除"):
        count = len(st.session_state.custom_excluded_words)
        st.session_state.custom_excluded_words = set()
        st.success(f"除外単語をすべて削除しました（{count}個）。")
        st.rerun()

# 結果表示設定
st.sidebar.header("結果表示設定")

show_scores = st.sidebar.checkbox(
    "類似度を表示",
    value=False,
    help="各トークンの下に類似度スコアを縦に並べて表示します。上段が表記類似度、下段が意味類似度です（いずれも0.000-1.000）。"
)

show_top_words = st.sidebar.checkbox(
    "上位単語を表示",
    value=False,
    help="各類似度手法について、スコア順に全単語を表示します（スクロールで全て確認可能）。"
)


# メインエリア
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("OCRテキスト入力")
    text_input = st.text_area(
        "テキストを入力してください",
        height=400,
        placeholder="ここにOCRで読み取ったテキストを貼り付けてください..."
    )

with col2:
    st.subheader("検索クエリ")
    query_input = st.text_input(
        "検索語を入力してください",
        placeholder="例: 重要"
    )


# 処理実行
if text_input and query_input:
    # 入力が変更されたかチェック
    input_changed = (
        st.session_state.last_text_input != text_input or 
        st.session_state.last_query_input != query_input
    )
    
    # 入力が変更されていない場合は計算をスキップ
    if not input_changed and st.session_state.last_tokens is not None and \
       st.session_state.string_scores is not None:
        # 既存の結果を使用（計算はスキップ）
        tokens = st.session_state.last_tokens
        string_scores = np.array(st.session_state.string_scores)
        semantic_scores = np.array(st.session_state.semantic_scores) if st.session_state.semantic_scores is not None else np.zeros(len(tokens))
        
        # ハイライト生成（既存の結果を使用）
        highlighted_html = highlight_with_string_and_semantic(
            tokens=tokens,
            string_scores=string_scores,
            semantic_scores=semantic_scores,
            string_threshold=string_threshold,
            semantic_threshold=semantic_threshold,
            use_string=use_string,
            use_semantic=use_semantic,
            show_scores=show_scores,
            is_punctuation_func=lambda t: is_excluded_word(t, st.session_state.custom_excluded_words)
        )
        
        # 結果表示のみ（計算はスキップ）
        st.subheader("ハイライト結果")
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        # 上位単語表示（計算スキップ時も表示）
        if show_top_words:
            st.subheader("スコア上位単語")
            
            col_top1, col_top2 = st.columns(2)
            
            with col_top1:
                if use_string:
                    st.write("**表記類似度 上位単語**")
                    # 同じ単語をまとめて、最高スコアを取得
                    token_score_dict = {}
                    for i, token in enumerate(tokens):
                        # 記号は除外
                        if not is_excluded_word(token, st.session_state.custom_excluded_words):
                            if token not in token_score_dict:
                                token_score_dict[token] = string_scores[i]
                            else:
                                token_score_dict[token] = max(token_score_dict[token], string_scores[i])
                    
                    # スコアでソート（降順）- 全件表示
                    string_top_data = sorted(token_score_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    # テーブル形式で表示（同じスコアは同じ順位）
                    top_string_df = []
                    current_rank = 1
                    prev_score = None
                    for idx, (token, score) in enumerate(string_top_data):
                        # 前のスコアと異なる場合、新しい順位を設定
                        if prev_score is not None and abs(score - prev_score) > 0.01:  # 0.01の誤差を許容
                            current_rank = idx + 1
                        elif prev_score is None:
                            current_rank = 1
                        prev_score = score
                        
                        top_string_df.append({
                            "順位": current_rank,
                            "単語": token,
                            "スコア": f"{score:.3f}"
                        })
                    
                    if top_string_df:
                        st.dataframe(pd.DataFrame(top_string_df), use_container_width=True, hide_index=True)
                else:
                    st.write("表記類似は無効です")
            
            with col_top2:
                if use_semantic:
                    st.write("**意味類似度 上位単語**")
                    # 同じ単語をまとめて、最高スコアを取得
                    token_score_dict = {}
                    for i, token in enumerate(tokens):
                        # 記号は除外
                        if not is_excluded_word(token, st.session_state.custom_excluded_words):
                            if token not in token_score_dict:
                                token_score_dict[token] = semantic_scores[i]
                            else:
                                token_score_dict[token] = max(token_score_dict[token], semantic_scores[i])
                    
                    # スコアでソート（降順）- 全件表示
                    semantic_top_data = sorted(token_score_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    # テーブル形式で表示（同じスコアは同じ順位）
                    top_semantic_df = []
                    current_rank = 1
                    prev_score = None
                    for idx, (token, score) in enumerate(semantic_top_data):
                        # 前のスコアと異なる場合、新しい順位を設定
                        if prev_score is not None and abs(score - prev_score) > 0.0001:  # 0.0001の誤差を許容
                            current_rank = idx + 1
                        elif prev_score is None:
                            current_rank = 1
                        prev_score = score
                        
                        top_semantic_df.append({
                            "順位": current_rank,
                            "単語": token,
                            "スコア": f"{score:.3f}"
                        })
                    
                    if top_semantic_df:
                        st.dataframe(pd.DataFrame(top_semantic_df), use_container_width=True, hide_index=True)
                else:
                    st.write("意味類似は無効です")
    else:
        # 入力が変更された場合のみ計算を実行
        # 入力が変更された場合のみ計算を実行
        # 進捗表示用のコンテナを作成
        progress_container = st.container()
        status_text = progress_container.empty()
        progress_bar = progress_container.progress(0)
        time_text = progress_container.empty()
        
        start_time = time.time()
        last_update_time = [start_time]  # リストで可変にする
        current_progress_value = [0.0]  # 現在の進捗値を追跡（リストで可変にする）
        
        # タイマー更新ヘルパー関数
        def update_time():
            current_time = time.time()
            # 0.05秒ごとに更新
            if current_time - last_update_time[0] >= 0.05:
                elapsed_time = current_time - start_time
                time_text.text(f"経過時間: {elapsed_time:.2f}秒")
                last_update_time[0] = current_time
                time.sleep(0.01)  # Streamlitに更新の機会を与える
        
        # 進捗更新ヘルパー関数（全体の進捗を0〜1で表示）
        def update_progress(current_progress, status_msg="処理中..."):
            current_time = time.time()
            # 0.05秒ごとに更新（または進捗が変更された場合）
            if current_time - last_update_time[0] >= 0.05 or current_progress != current_progress_value[0]:
                elapsed_time = current_time - start_time
                progress_bar.progress(current_progress)
                status_text.text(status_msg)
                time_text.text(f"経過時間: {elapsed_time:.2f}秒 ({int(current_progress * 100)}%)")
                last_update_time[0] = current_time
                current_progress_value[0] = current_progress
                time.sleep(0.01)  # Streamlitに更新の機会を与える
            else:
                # 進捗が変わらなくても、タイマーだけは更新
                update_time()
        
        # トークン化（キャッシュ確認）
        update_progress(0.1, "トークン化中...")
        
        tokens = st.session_state.token_cache.get(text_input)
        if tokens is None:
            tokens = tokenize(text_input)
            st.session_state.token_cache.set(text_input, tokens)
        
        if not tokens:
            st.warning("トークンが見つかりませんでした。")
        else:
            # 文字列類似度計算
            update_progress(0.2, "文字列類似度を計算中...")
            
            string_sim = StringSimilarity()
            # トークンごとに処理し、進捗を更新
            string_scores_list = []
            total_tokens = len(tokens)
            for i, token in enumerate(tokens):
                # 記号の場合はスコアを0にする
                if is_excluded_word(token, st.session_state.custom_excluded_words):
                    score = 0.0
                else:
                    score = string_sim.similarity(query_input, token)
                string_scores_list.append(score)
                
                # 進捗を計算（0.2〜0.25の範囲）
                progress = 0.2 + (0.05 * (i + 1) / total_tokens) if total_tokens > 0 else 0.25
                progress = min(0.25, progress)
                
                # 毎回タイマーを更新（0.05秒ごとに自動更新される）
                update_progress(progress, f"文字列類似度を計算中... ({i+1}/{total_tokens})")
            
            string_scores = np.array(string_scores_list)
            # 表記類似度を0〜1のスケールに正規化（100で割る）
            string_scores = string_scores / 100.0
            
            # 進捗更新
            update_progress(0.25, "文字列類似度計算完了")
            
            # 意味類似度計算（キャッシュ確認）
            semantic_scores = None
            if use_semantic:
                semantic_sim = st.session_state.semantic_similarity
                
                # トークンの埋め込み（キャッシュ確認）
                token_embeddings = st.session_state.embedding_cache.get(text_input)
                if token_embeddings is None:
                    # 初回またはテキストが変更された場合
                    update_progress(0.4, "トークンの埋め込みを計算中...（初回のみ時間がかかります）")
                    
                    # 埋め込み計算をバッチに分けて、各バッチの間にタイマーを更新
                    # バッチサイズを小さくして更新頻度を上げる（20-30バッチ程度に分割）
                    batch_size = max(5, len(tokens) // 25)  # 25バッチ程度に分割
                    if batch_size == 0:
                        batch_size = len(tokens)
                    
                    embeddings_list = []
                    total_batches = (len(tokens) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(0, len(tokens), batch_size):
                        batch_tokens = tokens[batch_idx:batch_idx + batch_size]
                        
                        # バッチ処理前の進捗を計算
                        progress = 0.4 + (0.3 * batch_idx / len(tokens))
                        progress = min(0.7, progress)
                        batch_num = (batch_idx // batch_size) + 1
                        
                        # バッチ処理前にタイマーを更新（強制的に更新）
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        progress_bar.progress(progress)
                        status_text.text(f"トークンの埋め込みを計算中... ({batch_num}/{total_batches}バッチ)")
                        time_text.text(f"経過時間: {elapsed_time:.2f}秒 ({int(progress * 100)}%)")
                        last_update_time[0] = current_time
                        current_progress_value[0] = progress
                        time.sleep(0.01)  # Streamlitに更新の機会を与える
                        
                        # バッチ処理（この間はタイマーを更新できない）
                        # 記号の埋め込みも計算する（後の処理で意味類似度を0にする）
                        batch_embeddings = semantic_sim.encode(batch_tokens)
                        embeddings_list.append(batch_embeddings)
                        
                        # バッチ処理後の進捗を計算
                        progress = 0.4 + (0.3 * (batch_idx + len(batch_tokens)) / len(tokens))
                        progress = min(0.7, progress)
                        
                        # バッチ処理後にタイマーを更新（進捗は更新されるが、タイマーも確実に更新）
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        progress_bar.progress(progress)
                        status_text.text(f"トークンの埋め込みを計算中... ({batch_num}/{total_batches}バッチ完了)")
                        time_text.text(f"経過時間: {elapsed_time:.2f}秒 ({int(progress * 100)}%)")
                        last_update_time[0] = current_time
                        current_progress_value[0] = progress
                        time.sleep(0.01)  # Streamlitに更新の機会を与える
                    
                    # すべてのバッチを結合
                    embeddings = np.vstack(embeddings_list)
                    
                    # 計算完了後にタイマーを更新
                    update_progress(0.7, "トークンの埋め込み計算完了")
                    
                    token_embeddings = embeddings.tolist()
                    st.session_state.embedding_cache.set(text_input, token_embeddings)
                else:
                    update_progress(0.5, "トークンの埋め込みを読み込み中...")
                    embeddings = np.array(token_embeddings)
                    update_progress(0.5, "トークンの埋め込み読み込み完了")
                
                # クエリの埋め込み
                update_progress(0.8, "クエリの埋め込みを計算中...")
                
                query_embedding = semantic_sim.encode([query_input])[0]
                
                update_progress(0.85, "クエリの埋め込み計算完了")
                
                # コサイン類似度計算
                update_progress(0.85, "意味類似度を計算中...")
                
                semantic_scores = semantic_sim.similarity(query_embedding, embeddings)
                
                # 記号の場合はスコアを0にする
                for i, token in enumerate(tokens):
                    if is_excluded_word(token, st.session_state.custom_excluded_words):
                        semantic_scores[i] = 0.0
                
                update_progress(0.90, "意味類似度計算完了")
            else:
                update_progress(0.9, "意味類似度はスキップされています")
                semantic_scores = np.zeros(len(tokens))
            
            # スコアをセッション状態に保存（自動設定用）
            st.session_state.string_scores = string_scores.tolist() if isinstance(string_scores, np.ndarray) else string_scores
            st.session_state.semantic_scores = semantic_scores.tolist() if isinstance(semantic_scores, np.ndarray) else semantic_scores
            st.session_state.last_tokens = tokens
            st.session_state.last_text_input = text_input
            st.session_state.last_query_input = query_input
            
            # 自動設定の場合、計算されたスコアから閾値を再計算
            if threshold_input_method == "自動設定":
                # ユニークな単語ごとの最高スコアから閾値を再計算
                if use_string and len(string_scores) > 0:
                    token_score_dict = get_unique_word_scores(
                        tokens,
                        string_scores,
                        st.session_state.custom_excluded_words
                    )
                    if token_score_dict:
                        sorted_string_scores = sorted(token_score_dict.values(), reverse=True)
                        if len(sorted_string_scores) >= auto_highlight_count:
                            string_threshold = float(sorted_string_scores[auto_highlight_count - 1])
                        else:
                            string_threshold = float(sorted_string_scores[-1]) if len(sorted_string_scores) > 0 else 0.3
                
                if use_semantic and len(semantic_scores) > 0:
                    token_score_dict = get_unique_word_scores(
                        tokens,
                        semantic_scores,
                        st.session_state.custom_excluded_words
                    )
                    if token_score_dict:
                        sorted_semantic_scores = sorted(token_score_dict.values(), reverse=True)
                        if len(sorted_semantic_scores) >= auto_highlight_count:
                            semantic_threshold = float(sorted_semantic_scores[auto_highlight_count - 1])
                        else:
                            semantic_threshold = float(sorted_semantic_scores[-1]) if len(sorted_semantic_scores) > 0 else 0.75
            
            # ハイライト生成
            update_progress(0.95, "ハイライトを生成中...")
            
            highlighted_html = highlight_with_string_and_semantic(
                tokens=tokens,
                string_scores=string_scores,
                semantic_scores=semantic_scores,
                string_threshold=string_threshold,
                semantic_threshold=semantic_threshold,
                use_string=use_string,
                use_semantic=use_semantic,
                show_scores=show_scores,
                is_punctuation_func=lambda t: is_excluded_word(t, st.session_state.custom_excluded_words)
            )
            
            # 完了
            update_progress(1.0, "処理完了")
            elapsed_time = time.time() - start_time
            time_text.text(f"総処理時間: {elapsed_time:.2f}秒 (100%)")
            
            # 結果表示
            st.subheader("ハイライト結果")
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # 上位単語表示
            if show_top_words:
                st.subheader("スコア上位単語")
                
                col_top1, col_top2 = st.columns(2)
                
                with col_top1:
                    if use_string:
                        st.write("**表記類似度**")
                        # 同じ単語をまとめて、最高スコアを取得
                        token_score_dict = {}
                        for i, token in enumerate(tokens):
                            # 記号は除外
                            if not is_excluded_word(token, st.session_state.custom_excluded_words):
                                if token not in token_score_dict:
                                    token_score_dict[token] = string_scores[i]
                                else:
                                    token_score_dict[token] = max(token_score_dict[token], string_scores[i])
                        
                        # スコアでソート（降順）- 全件表示
                        string_top_data = sorted(token_score_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        # テーブル形式で表示（同じスコアは同じ順位）
                        top_string_df = []
                        current_rank = 1
                        prev_score = None
                        for idx, (token, score) in enumerate(string_top_data):
                            # 前のスコアと異なる場合、新しい順位を設定
                            if prev_score is not None and abs(score - prev_score) > 0.01:  # 0.01の誤差を許容
                                current_rank = idx + 1
                            elif prev_score is None:
                                current_rank = 1
                            prev_score = score
                            
                            top_string_df.append({
                                "順位": current_rank,
                                "単語": token,
                                "スコア": f"{score:.3f}"
                            })
                        
                        if top_string_df:
                            st.dataframe(pd.DataFrame(top_string_df), use_container_width=True, hide_index=True)
                    else:
                        st.write("表記類似は無効です")
                
                with col_top2:
                    if use_semantic:
                        st.write("**意味類似度**")
                        # 同じ単語をまとめて、最高スコアを取得
                        token_score_dict = {}
                        for i, token in enumerate(tokens):
                            # 記号は除外
                            if not is_excluded_word(token, st.session_state.custom_excluded_words):
                                if token not in token_score_dict:
                                    token_score_dict[token] = semantic_scores[i]
                                else:
                                    token_score_dict[token] = max(token_score_dict[token], semantic_scores[i])
                        
                        # スコアでソート（降順）- 全件表示
                        semantic_top_data = sorted(token_score_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        # テーブル形式で表示（同じスコアは同じ順位）
                        top_semantic_df = []
                        current_rank = 1
                        prev_score = None
                        for idx, (token, score) in enumerate(semantic_top_data):
                            # 前のスコアと異なる場合、新しい順位を設定
                            if prev_score is not None and abs(score - prev_score) > 0.0001:  # 0.0001の誤差を許容
                                current_rank = idx + 1
                            elif prev_score is None:
                                current_rank = 1
                            prev_score = score
                            
                            top_semantic_df.append({
                                "順位": current_rank,
                                "単語": token,
                                "スコア": f"{score:.3f}"
                            })
                        
                        if top_semantic_df:
                            st.dataframe(pd.DataFrame(top_semantic_df), use_container_width=True, hide_index=True)
                    else:
                        st.write("意味類似は無効です")

elif text_input and not query_input:
    st.info("検索語を入力してください。")
elif not text_input and query_input:
    st.info("OCRテキストを入力してください。")
else:
    st.info("左側にOCRテキスト、右上に検索語を入力してください。")
