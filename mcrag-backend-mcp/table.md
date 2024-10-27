# テーブル定義書

## テーブル: `tasks`

| カラム名           | データ型  | NULL許可 | 主キー | 外部キー | 説明                          |
| ------------------ | --------- | -------- | ------ | -------- | ----------------------------- |
| `task_id`          | INTEGER   | NOT NULL | YES    |          | タスクのID（自動増分）         |
| `task_name`        | TEXT      | NOT NULL |        |          | タスク名（ユニーク）           |
| `character_prompt` | TEXT      |          |        |          | キャラクターのプロンプト       |
| `user_prompt`      | TEXT      |          |        |          | ユーザー設定のプロンプト       |

---

## テーブル: `words_info`

| カラム名        | データ型  | NULL許可 | 主キー | 外部キー          | 説明                          |
| --------------- | --------- | -------- | ------ | ----------------- | ----------------------------- |
| `word_info_id`  | INTEGER   | NOT NULL | YES    |                   | ワード情報のID（自動増分）     |
| `task_id`       | INTEGER   |          |        | `tasks(task_id)`   | 関連するタスクのID             |
| `word`          | TEXT      |          |        |                   | ワード                         |
| `info`          | TEXT      |          |        |                   | ワードに関する情報             |

---

## テーブル: `conversations`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                      | 説明                            |
| ----------------- | --------- | -------- | ------ | ----------------------------- | ------------------------------- |
| `conversation_id` | INTEGER   | NOT NULL | YES    |                               | 会話のID（自動増分）             |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`              | 関連するタスクのID               |
| `word_info_id`    | INTEGER   | NOT NULL |        | `words_info(word_info_id)`    | 関連するワード情報のID           |
| `talk_num`        | TEXT      | NOT NULL |        |                               | 会話の番号                       |
| `user`            | TEXT      |          |        |                               | ユーザーの発言内容               |
| `assistant`       | TEXT      |          |        |                               | アシスタントの発言内容           |
| `token_count`     | TEXT      |          |        |                               | トークン数                       |
| `processing_time` | TEXT      |          |        |                               | 処理時間                         |
| `temperature`     | TEXT      |          |        |                               | 温度パラメータ                   |
| `created_at`      | TEXT      |          |        |                               | 作成日時                         |

---

## テーブル: `generated_qas`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                     | 説明                            |
| ----------------- | --------- | -------- | ------ | ---------------------------- | ------------------------------- |
| `qa_id`           | INTEGER   | NOT NULL | YES    |                              | Q&AのID（自動増分）              |
| `task_name`       | TEXT      | NOT NULL |        |                              | タスク名                         |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`             | 関連するタスクのID               |
| `word_info_id`    | INTEGER   | NOT NULL |        | `words_info(word_info_id)`   | 関連するワード情報のID           |
| `talk_nums`       | TEXT      | NOT NULL |        |                              | 関連する会話の番号（複数可）      |
| `question`        | TEXT      |          |        |                              | 生成された質問                   |
| `answer`          | TEXT      |          |        |                              | 生成された回答                   |
| `token_count`     | INTEGER   |          |        |                              | トークン数                       |
| `processing_time` | TEXT      |          |        |                              | 処理時間                         |
| `created_at`      | TEXT      |          |        |                              | 作成日時                         |

---

## テーブル: `evaluated_answers`

| カラム名            | データ型  | NULL許可 | 主キー | 外部キー                     | 説明                            |
| ------------------- | --------- | -------- | ------ | ---------------------------- | ------------------------------- |
| `eval_id`           | INTEGER   | NOT NULL | YES    |                              | 評価のID（自動増分）             |
| `task_name`         | TEXT      | NOT NULL |        |                              | タスク名                         |
| `task_id`           | INTEGER   | NOT NULL |        | `tasks(task_id)`             | 関連するタスクのID               |
| `word_info_id`      | INTEGER   | NOT NULL |        | `words_info(word_info_id)`   | 関連するワード情報のID           |
| `talk_nums`         | TEXT      | NOT NULL |        |                              | 関連する会話の番号（複数可）      |
| `query`             | TEXT      |          |        |                              | 質問内容                         |
| `expected_answer`   | TEXT      |          |        |                              | 期待される回答                   |
| `gpt_response`      | TEXT      |          |        |                              | GPTの回答                        |
| `is_correct`        | INTEGER   |          |        |                              | 正誤判定（1:正しい、0:正しくない）|
| `evaluation_detail` | TEXT      |          |        |                              | 評価の詳細                       |
| `token_count`       | INTEGER   |          |        |                              | トークン数                       |
| `processing_time`   | TEXT       |          |        |                              | 処理時間                         |
| `model`             | TEXT      | NOT NULL |        |                              | 使用したモデル名                 |
| `created_at`        | TEXT      |          |        |                              | 作成日時                         |

---

## テーブル: `vector_table`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                           | 説明                           |
| ----------------- | --------- | -------- | ------ | ---------------------------------- | ------------------------------ |
| `vector_id`       | INTEGER   | NOT NULL | YES    |                                    | ベクトルのID（自動増分）        |
| `conversation_id` | INTEGER   | NOT NULL |        | `conversations(conversation_id)`   | 関連する会話のID                |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`                   | 関連するタスクのID              |
| `content`         | TEXT      |          |        |                                    | 会話の内容                      |
| `vector`          | BLOB      |          |        |                                    | ベクトルデータ                  |

---

## 制約

- **ユニーク制約**:
  - `tasks` テーブルの `task_name` はユニークです。
  - `conversations` テーブルの `(task_id, word_info_id, talk_num)` の組み合わせはユニークです。
  - `generated_qas` テーブルの `(task_name, word_info_id, talk_nums)` の組み合わせはユニークです。
  - `evaluated_answers` テーブルの `(task_name, word_info_id, talk_nums, model)` の組み合わせはユニークです。

- **外部キー制約**:
  - `words_info.task_id` は `tasks(task_id)` を参照します。
  - `conversations.task_id` は `tasks(task_id)` を参照します。
  - `conversations.word_info_id` は `words_info(word_info_id)` を参照します。
  - `generated_qas.task_id` は `tasks(task_id)` を参照します。
  - `generated_qas.word_info_id` は `words_info(word_info_id)` を参照します。
  - `evaluated_answers.task_id` は `tasks(task_id)` を参照します。
  - `evaluated_answers.word_info_id` は `words_info(word_info_id)` を参照します。
  - `vector_table.conversation_id` は `conversations(conversation_id)` を参照します。
  - `vector_table.task_id` は `tasks(task_id)` を参照します。

---

