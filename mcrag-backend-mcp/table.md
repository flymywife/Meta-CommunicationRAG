# テーブル定義書

## テーブル: `tasks`

| カラム名           | データ型  | NULL許可 | 主キー | 外部キー | 説明                          |
| ------------------ | --------- | -------- | ------ | -------- | ----------------------------- |
| `task_id`          | INTEGER   | NOT NULL | YES    |          | タスクのID（自動増分）         |
| `task_name`        | TEXT      | NOT NULL |        |          | タスク名（ユニーク）           |
| `character_prompt` | TEXT      |          |        |          | キャラクターのプロンプト       |
| `user_prompt`      | TEXT      |          |        |          | ユーザー設定のプロンプト       |

- **ユニーク制約**  
  - `task_name`
- **外部キー制約**  
  - なし

---

## テーブル: `words_info`

| カラム名       | データ型  | NULL許可 | 主キー | 外部キー          | 説明                       |
| -------------- | --------- | -------- | ------ | ----------------- | -------------------------- |
| `word_info_id` | INTEGER   | NOT NULL | YES    |                   | ワード情報のID（自動増分）  |
| `task_id`      | INTEGER   |          |        | `tasks(task_id)`  | 関連するタスクのID          |
| `word`         | TEXT      |          |        |                   | ワード                      |
| `info`         | TEXT      |          |        |                   | ワードに関する情報          |

- **外部キー制約**  
  - `words_info.task_id` → `tasks(task_id)`

---

## テーブル: `conversations`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                      | 説明                                           |
| ----------------- | --------- | -------- | ------ | ---------------------------- | ---------------------------------------------- |
| `conversation_id` | INTEGER   | NOT NULL | YES    |                              | 会話のID（自動増分）                            |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`             | 関連するタスクのID                              |
| `word_info_id`    | INTEGER   | NOT NULL |        | `words_info(word_info_id)`   | 関連するワード情報のID                          |
| `talk_num`        | TEXT      | NOT NULL |        |                              | 会話の番号                                      |
| `user`            | TEXT      |          |        |                              | ユーザーの発言内容                              |
| `assistant`       | TEXT      |          |        |                              | アシスタントの発言内容                          |
| `before_user`     | TEXT      |          |        |                              | 置換前のユーザー発言内容（コード側追加）         |
| `before_assistant`| TEXT      |          |        |                              | 置換のアシスタント発言内容（コード側追加）     |
| `input_token`     | INTEGER   |          |        |                              | 入力トークン数（コード側追加）                  |
| `output_token`    | INTEGER   |          |        |                              | 出力トークン数（コード側追加）                  |
| `processing_time` | TEXT      |          |        |                              | 処理時間                                        |
| `temperature`     | TEXT      |          |        |                              | temparatureパラメータ                                  |
| `created_at`      | TEXT      |          |        |                              | 作成日時                                        |

- **ユニーク制約**  
  - `(task_id, word_info_id, talk_num)`
- **外部キー制約**  
  - `conversations.task_id` → `tasks(task_id)`  
  - `conversations.word_info_id` → `words_info(word_info_id)`

---

## テーブル: `generated_qas`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                     | 説明                                 |
| ----------------- | --------- | -------- | ------ | ---------------------------- | ------------------------------------ |
| `qa_id`           | INTEGER   | NOT NULL | YES    |                              | Q&AのID（自動増分）                    |
| `task_name`       | TEXT      | NOT NULL |        |                              | タスク名                               |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`             | 関連するタスクのID                     |
| `word_info_id`    | INTEGER   | NOT NULL |        | `words_info(word_info_id)`   | 関連するワード情報のID                 |
| `talk_nums`       | TEXT      | NOT NULL |        |                              | 関連する会話の番号（複数可）            |
| `question`        | TEXT      |          |        |                              | 生成された質問                         |
| `answer`          | TEXT      |          |        |                              | 生成された回答                         |
| `input_token`     | INTEGER   |          |        |                              | 入力トークン数（コード側追加）         |
| `output_token`    | INTEGER   |          |        |                              | 出力トークン数（コード側追加）         |
| `processing_time` | REAL      |          |        |                              | 処理時間（コード側では REAL）          |
| `created_at`      | TEXT      |          |        |                              | 作成日時                               |

- **ユニーク制約**  
  - `(task_name, word_info_id, talk_nums)`
- **外部キー制約**  
  - `generated_qas.task_id` → `tasks(task_id)`  
  - `generated_qas.word_info_id` → `words_info(word_info_id)`

---

## テーブル: `evaluated_answers`

| カラム名            | データ型  | NULL許可 | 主キー | 外部キー                     | 説明                                             |
| ------------------- | --------- | -------- | ------ | --------------------------- | ------------------------------------------------ |
| `eval_id`           | INTEGER   | NOT NULL | YES    |                              | 評価のID（自動増分）                              |
| `qa_id`             | INTEGER   | NOT NULL |        |                              | 関連するQ&AのID（コード側追加）                   |
| `task_name`         | TEXT      | NOT NULL |        |                              | タスク名                                         |
| `task_id`           | INTEGER   | NOT NULL |        | `tasks(task_id)`             | 関連するタスクのID                               |
| `word_info_id`      | INTEGER   | NOT NULL |        | `words_info(word_info_id)`   | 関連するワード情報のID                           |
| `talk_nums`         | TEXT      | NOT NULL |        |                              | 関連する会話の番号（複数可）                      |
| `question`          | TEXT      |          |        |                              | 質問内容                                         |
| `expected_answer`   | TEXT      |          |        |                              | 期待される回答                                   |
| `gpt_response`      | TEXT      |          |        |                              | GPTの回答                                        |
| `get_context`       | TEXT      |          |        |                              | コンテキスト（コード側追加）                     |
| `get_talk_nums`     | TEXT      |          |        |                              | 使用した会話の番号（コード側追加）               |
| `input_token`       | INTEGER   |          |        |                              | 入力トークン数（コード側追加）                   |
| `output_token`      | INTEGER   |          |        |                              | 出力トークン数（コード側追加）                   |
| `processing_time`   | REAL      |          |        |                              | 処理時間（コード側では REAL）                    |
| `model`             | TEXT      | NOT NULL |        |                              | 使用したモデル名                                 |
| `created_at`        | TEXT      |          |        |                              | 作成日時                                         |

- **ユニーク制約**  
  - `(task_name, word_info_id, talk_nums, model)`
- **外部キー制約**  
  - `evaluated_answers.task_id` → `tasks(task_id)`  
  - `evaluated_answers.word_info_id` → `words_info(word_info_id)`

---

## テーブル: `vector_table`

| カラム名       | データ型  | NULL許可 | 主キー | 外部キー                       | 説明                                    |
| -------------- | --------- | -------- | ------ | ----------------------------- | --------------------------------------- |
| `vector_id`    | INTEGER   | NOT NULL | YES    |                               | ベクトルのID（自動増分）                 |
| `task_id`      | INTEGER   | NOT NULL |        | `tasks(task_id)`              | 関連するタスクのID                       |
| `word_info_id` | INTEGER   | NOT NULL |        |                               | 関連するワード情報のID（コード側でFK定義なし） |
| `talk_num`     | TEXT      | NOT NULL |        |                               | 会話の番号                               |
| `content`      | TEXT      |          |        |                               | 会話の内容                               |
| `row_vector`   | BLOB      |          |        |                               | 元のベクトルデータ（コード側追加）       |
| `vector`       | BLOB      |          |        |                               | 加工後のベクトルデータ（コード側追加）   |
| `pca_vector`   | BLOB      |          |        |                               | PCA後のベクトルデータ（コード側追加）    |

- **外部キー制約**  
  - `vector_table.task_id` → `tasks(task_id)`  
  - ※ `vector_table.word_info_id` はコード上で外部キー制約を定義していません

---

## テーブル: `rag_results`

| カラム名             | データ型  | NULL許可 | 主キー | 外部キー                       | 説明                                            |
| -------------------- | --------- | -------- | ------ | ----------------------------- | ----------------------------------------------- |
| `rag_id`             | INTEGER   | NOT NULL | YES    |                               | RAG結果のID（自動増分）                         |
| `qa_id`              | INTEGER   | NOT NULL |        | `generated_qas(qa_id)`        | 関連する Q&A のID                                |
| `task_name`          | TEXT      | NOT NULL |        |                               | タスク名                                        |
| `task_id`            | INTEGER   | NOT NULL |        | `tasks(task_id)`              | 関連するタスクのID                              |
| `talk_num_1`         | TEXT      | NOT NULL |        |                               | RAGが取得した会話番号1                        |
| `talk_num_2`         | TEXT      | NOT NULL |        |                               | RAGが取得した会話番号2                        |
| `talk_num_3`         | TEXT      | NOT NULL |        |                               | RAGが取得した会話番号3                        |
| `talk_num_4`         | TEXT      | NOT NULL |        |                               | RAGが取得した会話番号4                        |
| `talk_num_5`         | TEXT      | NOT NULL |        |                               | RAGが取得した会話番号5                        |
| `cosine_similarity_1`| REAL     |          |        |                               | コサイン類似度1                                 |
| `cosine_similarity_2`| REAL     |          |        |                               | コサイン類似度2                                 |
| `cosine_similarity_3`| REAL     |          |        |                               | コサイン類似度3                                 |
| `cosine_similarity_4`| REAL     |          |        |                               | コサイン類似度4                                 |
| `cosine_similarity_5`| REAL     |          |        |                               | コサイン類似度5                                 |
| `BM25_score_1`       | REAL     |          |        |                               | BM25スコア1                                     |
| `BM25_score_2`       | REAL     |          |        |                               | BM25スコア2                                     |
| `BM25_score_3`       | REAL     |          |        |                               | BM25スコア3                                     |
| `BM25_score_4`       | REAL     |          |        |                               | BM25スコア4                                     |
| `BM25_score_5`       | REAL     |          |        |                               | BM25スコア5                                     |
| `rss_rank_1`         | REAL     |          |        |                               | RSSランキング1                                 |
| `rss_rank_2`         | REAL     |          |        |                               | RSSランキング2                                 |
| `rss_rank_3`         | REAL     |          |        |                               | RSSランキング3                                 |
| `rss_rank_4`         | REAL     |          |        |                               | RSSランキング4                                 |
| `rss_rank_5`         | REAL     |          |        |                               | RSSランキング5                                 |
| `rerank_score_1`     | REAL     |          |        |                               | 再ランキングスコア1                             |
| `rerank_score_2`     | REAL     |          |        |                               | 再ランキングスコア2                             |
| `rerank_score_3`     | REAL     |          |        |                               | 再ランキングスコア3                             |
| `rerank_score_4`     | REAL     |          |        |                               | 再ランキングスコア4                             |
| `rerank_score_5`     | REAL     |          |        |                               | 再ランキングスコア5                             |
| `processing_time`    | REAL     | NOT NULL |        |                               | 処理時間（コード側では REAL）                  |
| `model`              | TEXT     | NOT NULL |        |                               | 使用したモデル名                                |
| `input_token_count`  | INTEGER  |          |        |                               | 入力トークン数（コード側追加）                 |
| `output_token_count` | INTEGER  |          |        |                               | 出力トークン数（コード側追加）                 |
| `subject`            | TEXT     |          |        |                               | 抽出した固有名詞                                    |
| `created_at`         | TEXT     | NOT NULL |        |                               | 作成日時                                        |

- **外部キー制約**  
  - `rag_results.task_id` → `tasks(task_id)`  
  - `rag_results.qa_id` → `generated_qas(qa_id)`

---

## テーブル: `svd_analysis`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                       | 説明                               |
| ----------------- | --------- | -------- | ------ | ----------------------------- | ---------------------------------- |
| `svd_id`          | INTEGER   | NOT NULL | YES    |                               | SVD解析のID（自動増分）             |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`              | 関連するタスクのID                  |
| `task_name`       | TEXT      | NOT NULL |        |                               | タスク名                            |
| `qa_id`           | INTEGER   | NOT NULL |        | `generated_qas(qa_id)`        | 関連するQ&AのID                     |
| `expected_vector` | BLOB      |          |        |                               | 期待されるベクトルデータ           |
| `actual_vector`   | BLOB      |          |        |                               | 実際のベクトルデータ               |
| `created_at`      | TEXT      |          |        |                               | 作成日時                            |

- **外部キー制約**  
  - `svd_analysis.task_id` → `tasks(task_id)`  
  - `svd_analysis.qa_id` → `generated_qas(qa_id)`

---

## テーブル: `pca_analysis`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                       | 説明                                     |
| ----------------- | --------- | -------- | ------ | ----------------------------- | ---------------------------------------- |
| `pca_id`          | INTEGER   | NOT NULL | YES    |                               | PCA解析のID（自動増分）                  |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`              | 関連するタスクのID                        |
| `task_name`       | TEXT      | NOT NULL |        |                               | タスク名                                  |
| `qa_id`           | INTEGER   | NOT NULL |        | `generated_qas(qa_id)`        | 関連するQ&AのID                           |
| `model`           | TEXT      | NOT NULL |        |                               | 使用したモデル名                          |
| `expected_vector` | BLOB      |          |        |                               | 期待されるベクトルデータ                 |
| `actual_vector`   | BLOB      |          |        |                               | 実際のベクトルデータ                     |
| `distance`        | REAL      |          |        |                               | ベクトル間の距離                         |
| `created_at`      | TEXT      |          |        |                               | 作成日時                                  |

- **外部キー制約**  
  - `pca_analysis.task_id` → `tasks(task_id)`  
  - `pca_analysis.qa_id` → `generated_qas(qa_id)`

---

## 制約まとめ

### ユニーク制約

- **`tasks` テーブル**  
  - `task_name`
- **`conversations` テーブル**  
  - `(task_id, word_info_id, talk_num)`
- **`generated_qas` テーブル**  
  - `(task_name, word_info_id, talk_nums)`
- **`evaluated_answers` テーブル**  
  - `(task_name, word_info_id, talk_nums, model)`

### 外部キー制約

- **`words_info` テーブル**  
  - `task_id` → `tasks(task_id)`
- **`conversations` テーブル**  
  - `task_id` → `tasks(task_id)`
  - `word_info_id` → `words_info(word_info_id)`
- **`generated_qas` テーブル**  
  - `task_id` → `tasks(task_id)`
  - `word_info_id` → `words_info(word_info_id)`
- **`evaluated_answers` テーブル**  
  - `task_id` → `tasks(task_id)`
  - `word_info_id` → `words_info(word_info_id)`
- **`vector_table` テーブル**  
  - `task_id` → `tasks(task_id)`  
  - ※ `word_info_id` の外部キー制約は未定義
- **`rag_results` テーブル**  
  - `task_id` → `tasks(task_id)`
  - `qa_id` → `generated_qas(qa_id)`
- **`svd_analysis` テーブル**  
  - `task_id` → `tasks(task_id)`
  - `qa_id` → `generated_qas(qa_id)`
- **`pca_analysis` テーブル**  
  - `task_id` → `tasks(task_id)`
  - `qa_id` → `generated_qas(qa_id)`
