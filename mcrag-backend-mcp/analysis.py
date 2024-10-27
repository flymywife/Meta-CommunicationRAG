# analysis.py

from database import ConversationDatabase

class Analysis:
    def __init__(self, db_file='conversation_data.db'):
        self.db = ConversationDatabase(db_file=db_file)
    
    def get_all_task_names(self):
        try:
            select_sql = 'SELECT DISTINCT task_name FROM tasks'
            self.db.cursor.execute(select_sql)
            rows = self.db.cursor.fetchall()
            task_names = [row[0] for row in rows]
            return task_names
        except Exception as e:
            print(f"Error fetching task names: {e}")
            return []
    
    def close(self):
        self.db.close()
