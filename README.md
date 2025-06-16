# Transformers ノートブック

このリポジトリには、O'Reilly書籍「[Natural Language Processing with Transformers](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/)」のサンプルコードが含まれています：

<img alt="book-cover" height=200 src="images/book_cover.jpg" id="book-cover"/>

## はじめに

これらのノートブックは[Google Colab](https://colab.research.google.com/)で実行できます。ほとんどの章では適切な実行時間でGPUが必要なため、CUDAが事前にインストールされているGoogle Colabの使用を推奨します。

### Google Colabでの実行

Google Colabでこれらのノートブックを実行するには、以下の表のバッジをクリックしてください：

<!--This table is automatically generated, do not fill manually!-->



| 章                                          | Colab                                                                                                                                                                                               |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| はじめに                                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/01_introduction.ipynb)              |
| テキスト分類                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/02_classification.ipynb)            |
| Transformerの構造                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/03_transformer-anatomy.ipynb)       |
| 多言語固有表現認識                          | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/04_multilingual-ner.ipynb)          |
| テキスト生成                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/05_text-generation.ipynb)           |
| 要約                                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/06_summarization.ipynb)             |
| 質問応答                                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/07_question-answering.ipynb)        |
| 本番環境でのTransformer効率化               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/08_model-compression.ipynb)         |
| 少数ラベル・無ラベル学習                    | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/09_few-to-no-labels.ipynb)          |
| Transformerのゼロからの訓練                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/10_transformers-from-scratch.ipynb) |
| 将来の方向性                                | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ohtani-y/notebooks/blob/main/11_future-directions.ipynb)         |

<!--End of table-->

> 注意: Google Colabでは、パッケージをインストールした後にランタイムを再起動する必要がある場合があります。

### ローカル環境での実行

ローカル環境でノートブックを実行するには、まずリポジトリをクローンして移動します：

```bash
$ git clone https://github.com/Ohtani-y/notebooks.git
$ cd notebooks
```

次に、以下のコマンドを実行して、ノートブックの実行に必要なすべてのライブラリを含む`conda`仮想環境を作成します：

```bash
$ conda env create -f environment.yml
```

> 注意: 環境を構築するには、NVIDIA の [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) をサポートするGPUが必要です。現在、これはApple siliconではローカルで構築できないことを意味します 😢。

第7章（質問応答）には特別な依存関係があるため、その章を実行するには別の環境が必要です：

```bash
$ conda env create -f environment-chapter7.yml
```

依存関係をインストールしたら、`conda`環境をアクティブ化してノートブックを起動できます：

```bash
$ conda activate book # または conda activate book-chapter7
$ jupyter notebook
```

## よくある質問

### Google Colabでノートブックを実行する際にGPUを有効にするにはどうすればよいですか？

Google Colabでは、「ランタイム」>「ランタイムのタイプを変更」を選択し、ハードウェアアクセラレータとしてGPUを選択することでGPUを有効にできます。

## 引用

この書籍を引用したい場合は、以下のBibTeXエントリを使用できます：

```
@book{tunstall2022natural,
  title={Natural Language Processing with Transformers: Building Language Applications with Hugging Face},
  author={Tunstall, Lewis and von Werra, Leandro and Wolf, Thomas},
  isbn={1098103246},
  url={https://books.google.ch/books?id=7hhyzgEACAAJ},
  year={2022},
  publisher={O'Reilly Media, Incorporated}
}
```
