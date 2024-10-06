# テーブル定義書

## テーブル: `tasks`

| カラム名            | データ型  | NULL許可 | 主キー | 外部キー | 説明                        |
| ------------------- | --------- | -------- | ------ | -------- | --------------------------- |
| `task_id`           | INTEGER   | NOT NULL | YES    |          | タスクのID（自動増分）       |
| `task_name`         | TEXT      | NOT NULL |        |          | タスク名                    |
| `character_prompt`  | TEXT      |          |        |          | キャラクタープロンプト       |
| `user_prompt`       | TEXT      |          |        |          | ユーザープロンプト           |

---

## テーブル: `words_info`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー        | 説明                         |
| ----------------- | --------- | -------- | ------ | -------------- | ---------------------------- |
| `word_info_id`    | INTEGER   | NOT NULL | YES    |                | ワード情報のID（自動増分）     |
| `task_id`         | INTEGER   |          |        | `tasks(task_id)`| 関連するタスクのID            |
| `word`            | TEXT      |          |        |                | ワード                        |
| `info`            | TEXT      |          |        |                | ワードに関する情報            |

---

## テーブル: `conversations`

| カラム名          | データ型  | NULL許可 | 主キー | 外部キー                | 説明                         |
| ----------------- | --------- | -------- | ------ | ----------------------- | ---------------------------- |
| `conversation_id` | INTEGER   | NOT NULL | YES    |                          | 会話のID（自動増分）           |
| `task_id`         | INTEGER   | NOT NULL |        | `tasks(task_id)`         | 関連するタスクのID            |
| `word_info_id`    | INTEGER   | NOT NULL |        | `words_info(word_info_id)`| 関連するワード情報のID        |
| `talk_num`        | TEXT      | NOT NULL |        |                          | 会話の番号                    |
| `user`            | TEXT      |          |        |                          | ユーザーの発言内容            |
| `assistant`       | TEXT      |          |        |                          | アシスタントの発言内容        |
| `token_count`     | TEXT      |          |        |                          | トークン数                     |
| `processing_time` | TEXT      |          |        |                          | 処理時間                      |
| `temperature`     | TEXT      |          |        |                          | AIの温度パラメータ            |
| `created_at`      | TEXT      |          |        |                          | 作成日時                      |

---

## 制約
- `conversations` テーブルの `task_id`, `word_info_id`, `talk_num` は複合一意制約です。
- `tasks` と `words_info`、および `conversations` テーブル間では、外部キー制約が設定されています。
