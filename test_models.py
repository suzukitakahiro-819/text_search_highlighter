"""
モデルの動作確認テストファイル

このスクリプトは、プロジェクトで使用されている以下のモデル・機能が
正常に動作しているかを確認します：

1. rapidfuzz - 文字列類似度計算
2. sentence-transformers - 意味類似度計算（intfloat/multilingual-e5-small）
3. トークン化機能
4. 類似度計算統合
5. ハイライト機能

実行方法:
    python test_models.py
"""
import sys
import traceback
from typing import List
import numpy as np

# カラフルな出力のための色コード
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """ヘッダーを表示"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text: str):
    """成功メッセージを表示"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """エラーメッセージを表示"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """情報メッセージを表示"""
    print(f"{Colors.OKCYAN}  {text}{Colors.ENDC}")


def print_warning(text: str):
    """警告メッセージを表示"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


# ============================================================================
# テスト1: トークン化機能のテスト
# ============================================================================
def test_tokenization():
    """トークン化機能をテスト"""
    print_header("テスト1: トークン化機能")
    
    try:
        from src.tokenize import tokenize
        
        # テストケース1: 日本語テキスト
        test_text1 = "人工知能技術の発展について"
        tokens1 = tokenize(test_text1)
        print_info(f"入力: {test_text1}")
        print_info(f"トークン数: {len(tokens1)}")
        print_info(f"トークン: {tokens1}")
        
        assert len(tokens1) > 0, "トークンが生成されませんでした"
        print_success(f"トークン化成功: {len(tokens1)}個のトークンを生成")
        
        # テストケース2: 日本語と英数字混合
        test_text2 = "AI技術は2024年に急速に発展しました。"
        tokens2 = tokenize(test_text2)
        print_info(f"入力: {test_text2}")
        print_info(f"トークン: {tokens2}")
        
        assert len(tokens2) > 0, "トークンが生成されませんでした"
        print_success(f"混合テキストのトークン化成功")
        
        return True
        
    except Exception as e:
        print_error(f"トークン化テスト失敗: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# テスト2: rapidfuzz（文字列類似度）のテスト
# ============================================================================
def test_rapidfuzz():
    """rapidfuzzによる文字列類似度計算をテスト"""
    print_header("テスト2: rapidfuzz（文字列類似度）")
    
    try:
        from src.similarity import StringSimilarity
        
        string_sim = StringSimilarity()
        
        # テストケース
        test_cases = [
            ("重要", "重要", 100.0),  # 完全一致
            ("重要", "重要な", 85.0),  # 部分一致（しきい値は大まか）
            ("重要", "緊急", 0.0),     # 不一致
        ]
        
        for query, token, expected_min in test_cases:
            score = string_sim.similarity(query, token)
            print_info(f"クエリ: '{query}' vs トークン: '{token}'")
            print_info(f"  類似度: {score:.1f} (期待値: {expected_min}以上)")
            
            assert 0 <= score <= 100, f"類似度は0〜100の範囲である必要があります: {score}"
            
            if expected_min > 0:
                assert score >= expected_min * 0.8, f"類似度が低すぎます: {score} < {expected_min * 0.8}"
        
        print_success("rapidfuzzによる文字列類似度計算が正常に動作しています")
        return True
        
    except Exception as e:
        print_error(f"rapidfuzzテスト失敗: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# テスト3: sentence-transformers（意味類似度）のテスト
# ============================================================================
def test_sentence_transformers():
    """sentence-transformersによる意味類似度計算をテスト"""
    print_header("テスト3: sentence-transformers（意味類似度）")
    
    try:
        from src.similarity import SemanticSimilarity
        
        print_info("モデルの読み込みを開始します...")
        print_warning("初回はモデルダウンロードに数分かかる場合があります")
        
        semantic_sim = SemanticSimilarity()
        
        # モデルが読み込まれたか確認（encodeを呼び出すと自動的に読み込まれる）
        test_texts = ["重要", "緊急", "必須"]
        embeddings = semantic_sim.encode(test_texts)
        
        print_info(f"モデル名: {semantic_sim.model_name}")
        print_info(f"埋め込みベクトルの形状: {embeddings.shape}")
        print_info(f"埋め込み次元数: {embeddings.shape[1]}")
        
        assert embeddings.shape[0] == len(test_texts), "埋め込みベクトルの数が一致しません"
        assert embeddings.shape[1] > 0, "埋め込み次元数が0です"
        
        print_success("モデルの読み込みと埋め込み計算が成功しました")
        
        # 類似度計算のテスト
        query_text = "重要"
        query_embedding = semantic_sim.encode([query_text])[0]
        
        similarities = semantic_sim.similarity(query_embedding, embeddings)
        
        print_info(f"クエリ: '{query_text}'")
        for text, sim in zip(test_texts, similarities):
            print_info(f"  vs '{text}': {sim:.3f}")
        
        assert len(similarities) == len(test_texts), "類似度の数が一致しません"
        assert all(0 <= s <= 1 for s in similarities), "類似度は0〜1の範囲である必要があります"
        
        # クエリと同一のテキストが最も高い類似度を持つことを確認
        query_idx = test_texts.index(query_text)
        assert similarities[query_idx] > 0.9, f"同一テキストの類似度が低すぎます: {similarities[query_idx]}"
        
        print_success("意味類似度計算が正常に動作しています")
        return True
        
    except ImportError as e:
        print_error(f"sentence-transformersがインストールされていない可能性があります: {str(e)}")
        print_warning("以下のコマンドでインストールしてください: pip install sentence-transformers")
        return False
    except Exception as e:
        print_error(f"sentence-transformersテスト失敗: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# テスト4: 統合テスト（実際の使用例）
# ============================================================================
def test_integration():
    """実際の使用例で統合テスト"""
    print_header("テスト4: 統合テスト（実際の使用例）")
    
    try:
        from src.tokenize import tokenize
        from src.similarity import StringSimilarity, SemanticSimilarity
        from src.highlight import highlight_with_string_and_semantic
        
        # テストテキスト
        test_text = "人工知能技術の発展について。近年、機械学習や深層学習の分野では、大規模なデータセットを活用したモデルの訓練が可能となりました。"
        query = "重要"
        
        print_info(f"テストテキスト: {test_text[:50]}...")
        print_info(f"検索クエリ: '{query}'")
        
        # トークン化
        tokens = tokenize(test_text)
        print_info(f"トークン数: {len(tokens)}")
        print_info(f"トークン例: {tokens[:5]}...")
        
        assert len(tokens) > 0, "トークンが生成されませんでした"
        
        # 文字列類似度計算
        string_sim = StringSimilarity()
        string_scores = np.array([string_sim.similarity(query, token) for token in tokens])
        
        print_info(f"文字列類似度の範囲: {string_scores.min():.1f} 〜 {string_scores.max():.1f}")
        
        # 意味類似度計算
        print_info("意味類似度を計算中...")
        semantic_sim = SemanticSimilarity()
        token_embeddings = semantic_sim.encode(tokens)
        query_embedding = semantic_sim.encode([query])[0]
        semantic_scores = semantic_sim.similarity(query_embedding, token_embeddings)
        
        print_info(f"意味類似度の範囲: {semantic_scores.min():.3f} 〜 {semantic_scores.max():.3f}")
        
        # ハイライト生成
        highlighted_html = highlight_with_string_and_semantic(
            tokens=tokens,
            string_scores=string_scores,
            semantic_scores=semantic_scores,
            string_threshold=30.0,
            semantic_threshold=0.75,
            use_string=True,
            use_semantic=True
        )
        
        # ハイライトされたトークン数をカウント
        highlighted_count = np.sum(
            (string_scores >= 30.0) | (semantic_scores >= 0.75)
        )
        
        print_info(f"ハイライトされたトークン数: {highlighted_count} / {len(tokens)}")
        print_info(f"HTML文字列の長さ: {len(highlighted_html)}文字")
        
        assert len(highlighted_html) > 0, "HTMLが生成されませんでした"
        
        print_success("統合テストが正常に完了しました")
        
        # 結果のプレビュー
        print_info("\n生成されたHTMLの最初の200文字:")
        print_info(highlighted_html[:200] + "...")
        
        return True
        
    except Exception as e:
        print_error(f"統合テスト失敗: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# テスト5: キャッシュ機能のテスト
# ============================================================================
def test_cache():
    """キャッシュ機能をテスト"""
    print_header("テスト5: キャッシュ機能")
    
    try:
        from src.cache import TokenCache, EmbeddingCache
        
        # TokenCacheのテスト
        token_cache = TokenCache()
        test_text = "テストテキスト"
        
        # 初回取得（キャッシュなし）
        result1 = token_cache.get(test_text)
        assert result1 is None, "初回はキャッシュがNoneであるべきです"
        print_success("TokenCache: 初回取得時にキャッシュが空であることを確認")
        
        # キャッシュに保存
        test_tokens = ["テスト", "テキスト"]
        token_cache.set(test_text, test_tokens)
        
        # 再度取得（キャッシュあり）
        result2 = token_cache.get(test_text)
        assert result2 == test_tokens, "キャッシュから正しいトークンが取得できませんでした"
        print_success("TokenCache: キャッシュからの取得が正常に動作しています")
        
        # EmbeddingCacheのテスト
        embedding_cache = EmbeddingCache()
        
        # 初回取得（キャッシュなし）
        result3 = embedding_cache.get(test_text)
        assert result3 is None, "初回はキャッシュがNoneであるべきです"
        print_success("EmbeddingCache: 初回取得時にキャッシュが空であることを確認")
        
        # キャッシュに保存
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        embedding_cache.set(test_text, test_embeddings)
        
        # 再度取得（キャッシュあり）
        result4 = embedding_cache.get(test_text)
        assert result4 == test_embeddings, "キャッシュから正しい埋め込みが取得できませんでした"
        print_success("EmbeddingCache: キャッシュからの取得が正常に動作しています")
        
        print_success("キャッシュ機能が正常に動作しています")
        return True
        
    except Exception as e:
        print_error(f"キャッシュテスト失敗: {str(e)}")
        traceback.print_exc()
        return False


# ============================================================================
# メイン実行部分
# ============================================================================
def main():
    """メイン関数"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}")
    print("=" * 60)
    print("  OCRテキスト検索ハイライター - モデル動作確認テスト")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    results = []
    
    # 各テストを実行
    results.append(("トークン化", test_tokenization()))
    results.append(("rapidfuzz", test_rapidfuzz()))
    results.append(("sentence-transformers", test_sentence_transformers()))
    results.append(("キャッシュ", test_cache()))
    results.append(("統合テスト", test_integration()))
    
    # 結果サマリー
    print_header("テスト結果サマリー")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print_success(f"{name}: 合格")
        else:
            print_error(f"{name}: 不合格")
    
    print(f"\n{Colors.BOLD}合計: {passed}/{total} テスト合格{Colors.ENDC}\n")
    
    if passed == total:
        print(f"{Colors.OKGREEN}{Colors.BOLD}✓ すべてのテストが正常に完了しました！{Colors.ENDC}\n")
        return 0
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}✗ 一部のテストが失敗しました。エラーメッセージを確認してください。{Colors.ENDC}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())