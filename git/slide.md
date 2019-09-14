<!-- 
$theme: uncover
template: invert
-->

# Git / GitHub

Yanny's Computer 山﨑祐太

<style scoped="scoped">
    h1 {
        text-align: center;
        font-size: 100px;
    }
    p {
        text-align: center;
        font-size: 40px;
    }
</style>

---
<!-- paginate: true -->

# 目的

本チュートリアルでは，以下の点について学んでいく

- Git / GitHubとは何か
- 基本的な使い方
- チーム開発におけるGitのTips

---

# Reference

- Gitのコマンドと解説一覧(https://git-scm.com/docs)
- Gitの全容を知る(https://git-scm.com/book/en/v2)

---

# Gitとは

![bg right 40%](https://git-scm.com/images/logos/downloads/Git-Logo-2Color.png)

- **分散型のバージョン管理ツール**
- 他にもバージョン管理ツールはあるが，現状**Gitの一択**
- ローカルとリモート(サーバー上)の両方にリポジトリをもつ
- ローカルで作業を行い変更履歴を更新し，リモートに変更を取り込む

---

# GitHub(GitLab)とは

- GitHubは**Gitをより便利に利用できるwebサービス**の名前
- チーム開発やコードの差分チェックなどが便利に！
- 同様のwebサービスにGitLabなどもある

![bg right 40%](https://github.githubassets.com/images/modules/logos_page/GitHub-Logo.png)

---

# なぜGitやGitHubを使うか

- **いつ誰が何を変更したか**がすぐに確認できる
- **過去のある時点にすぐ戻ることができる**
- 分散型バージョン管理では，**複数人での開発がより便利に**なる

---

# 基本的な使い方

1. リモートリポジトリをローカルにclone
2. 作業用のbranchを切る
3. ローカルで作業を行い，変更をcommit
4. ある程度commitが溜まったら(別にcommitは1つでも良い)push
5. pushされたbranchを元のbranchにmerge

---

![bg 20%](https://cdn.shopify.com/s/files/1/1061/1924/products/Thinking_Face_Emoji_large.png?v=1480481060)

---

# 1.リモートリポジトリをローカルにclone

```shell
git clone -b develop https://github.com/yutayamazaki/Tutorials.git
```

- GitHubなどからリポジトリをローカルに取ってくる

- `-b develop`でdevelopブランチを取ってくるという意味(ブランチは後で解説)

- リモートから取ってきて，ローカルで作業，リモートに変更を取り込むと言う流れになる

---

# 2. 作業用ブランチを切る

```shell
git checkout -b feature/fix_yamazaki
```

- `git checkout -b ブランチ名`でブランチを作成する
- とりあえず`feature/変更や機能の名前`というブランチを作成すればok
- 作業をcommitした後にこのブランチを元のブランチにマージする

---

# branch

- Gitで管理している履歴を枝分かれさせたもの
- 複数のブランチを作成してそれを本流に結合するという流れで開発する
- これにより複数人や複数チームが並行して別機能の開発を行える

---

# 3. ローカルで作業を行い，変更をcommit

```shell
git add yamazaki.md
git commit -m "add yamazaki.md"
```

- `git add file名`でステージングエリアに上げる
- `git commit -m "コミットメッセージ"`で変更を記録する

---

# commit

- Gitにおけるバージョン管理の単位
- ひとまとまりの作業を行うたびにcommitを行い，適宜変更を記録していく
- `git commit -m "コミットメッセージ"`でcommitする
- メッセージは`fix function`など「動詞+目的語」で何をどうしたかを書く
    - 参考記事：[GitHubで使われている実用英語コメント集](https://qiita.com/shikichee/items/a5f922a3ef3aa58a1839)
- commitの単位はひとまずは大きすぎなければok

---

# add

- Gitでcommit出来るのはステージングエリアにあるファイルだけ
- ステージングエリアがあることで，commitの単位を調整できる
- 以下の流れでcommitを行う
    1. ファイルに変更を加える
    2. ステージングエリアに追加
    3. commit

![bg right 80%](https://git-scm.com/figures/18333fig0106-tn.png)

---

# 4. ある程度commitが溜まったらpush

```shell
git push origin feature/fix_yamazaki
```

- 自分の作業ブランチをリモートに送る
- その後GitHubやGitLabでプルリクエストやマージリクエストを作成
- コードレビューや修正を経て元のブランチにマージされる

---

# GitのTips

<style scoped="scoped">
    h1 {
        text-align: center;
        font-size: 100px;
    }
</style>

---

# commitのTips

- commitはレビューの際に確認していく単位でもある
    - 大きすぎても小さすぎても面倒
    - **困ったときはより小さくまとめる(レビューしやすい)**

- **開発時にどんな変更が加えられたのかをコミットメッセージで確認**していく
    - メッセージは分かりやすく
    - commitが**意味のあるまとまり**だと理解しやすくなる！

---

# Pull Request(Merge Request)のTips

- GitHubではPull Request，GitLabではMerge Requestと呼ばれる
- 作業ブランチを元のブランチに統合する処理のこと
- Descriptionに何の変更を施したかを書く
    - 変更された機能だけでなく，その開発の背景や目的などがあるとなお良い

---

# 頻出Gitコマンド集

<style scoped="scoped">
    h1 {
        text-align: center;
        font-size: 80px;
    }
</style>

---

# 確認系のコマンド

#### ・現状確認
今いるブランチや変更されたファイルがステージングされているかなどを確認．

```shell
git status
```

#### ・直近数個のcommitのログを確認
```shell
git log
```

#### ・ステージング前のファイルの変更の差分を確認
追加した部分が緑，削除した分が赤で表示される．
```shell
git diff filename
```

---

# 作業取り消し系のコマンド

#### ・addの取り消し


```shell
git reset filename
```

#### ・commitの取り消し

```shell
git reset HEAD^
```

---

# Gitの流れを体験する

<style scoped="scoped">
    h1 {
        text-align: center;
        font-size: 80px;
    }
</style>

---

# やること

- https://github.com/yutayamazaki/Tutorials.git をforkしてローカルにcloneする
- 適当なファイルを作ってその変更をcommitする
- 自分のリモートリポジトリにpush
- 元のリモートリポジトリにPull Requestを送る
