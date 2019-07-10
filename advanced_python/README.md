# Advanced Python
## Index
### ch.1 Pythonコーディング一般
- Pythonで型を書く(Type Hints)
- コーディング規則(PEP8)に従う
- docstringを書く

### ch.2 関数とクラス
### ch.3 テスト

## ch.1 Pythonコーディング一般
### Pythonで型を書く(Type Hints)
Pythonは動的型付け言語で，基本的に型を書く必要はありません．  
ただしPython3.4からはType Hintsが導入され，関数の引数や戻り値の型アノテーションをつけることができるようになりました．  
型を明示することで，関数の使い方が明確になり，戻り値の型を把握するのに頭を悩ませることもありません．  

ちなみに型を書けるとはいえ，実際にそれ以外の型の値が入っても，エラーになることも，warningが出ることもありません．(Python自体の思想を打ち消す事になるため)  

以下のコード例を見てみましょう．  
textファイルを開いて，中身を取得する2つの関数です．  
- load_textファイルがtextファイルの中身全体を1つの文字列として
- load_text_as_listがtextファイルの中身を1行ずつ文字列のlistとして

読み込みます．  

関数の上にコメントを書けばもちろんどのような出力がなされるのかはわかりますが，Type Hintsを用いることでより行数が少なく，明示的に理解することができます．  

データ分析周りでも，numpy.ndarray，torch.Tensor，pandas.DataFrame(もしくはpandas.Series)なのかを把握したいことがよくあるので，Type Hintsは非常に良い助けになります．  

```python
def load_text(filename: str) -> str:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

def load_text_as_list(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        return f.readlines()
```


### コーディング規則(PEP8)に従う
- [PEP8-Ja](https://pep8-ja.readthedocs.io/ja/latest/)  
- [PEP8-En](https://www.python.org/dev/peps/pep-0008/)

同じ言語であっても人によって様々なコードの書き方をするので，複数人での開発の場合，プロジェクト全体のコードが，かなり混沌とした状態になることがよくあります．　　
そうなるとコード全体の見通しが悪くなり，読むのもうんざりするようなプログラムと向き合うことになります．　　
そうなることを避けるために，Pythonのコーディング規約として広く用いられるのがこのPEP8です．  

- レイアウト
- 式や文中の空白や改行
- 命名規約

など，様々な要素に関して規約を設けているので，特に何の断りもない場合はPEP8に従ったコーディングを心がけましょう．  
ただしPEP8が必ずしも全て正しいわけではない(実際import関連ではよく不満を耳にする)ので，チームとして規約を共有できるのであれば，チームやプロジェクトに独自のコーディング規約を設けても良いでしょう．  

### docstringを書く
docstringはただのコメントではなく，Pythonオブジェクトの，```__doc__```属性に文字列として格納されるコメントのことです．  
関数名やクラス名の直下に，"""と"""で囲まれたコメントを書くことで，実行時に関数やクラスから呼び出すことができます．また，Sphinxというドキュメント作成ツールを用いたときに，この```__doc__```属性から勝手にドキュメントを生成します．  

Pythonのdocstringには以下の2つの流派があります(NumPy Styleの方がよく見る)  
- Google Style
- NumPy Style

詳細は[こちら](http://www.sphinx-doc.org/ja/stable/ext/napoleon.html)を参照  

NumPy Styleで書いたdocstringの例をあげておきます．  

```python
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.falot64:
    """Calculate Accuracy Score for Classification.

    Parameters
    ----------
    y_true: np.ndarray
        1D-array, truth label
    y_pred: np.ndarray
        1D-array, predicted label

    Returns
    -------
    accuracy: np.float64
        Calculated accuracy score.
    """
    score = y_true == y_pred
    return np.mean(score)
```
