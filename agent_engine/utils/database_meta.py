import datetime
import decimal
from collections import defaultdict
from typing import Dict, Any

class DatabaseMetadata:
    def __init__(self, cursor, db_config):
        self.cursor = cursor
        self.db_config = db_config

    def _get_description(self, table_name: str) -> str:
        for table in self.db_config.meta.tables:
            if table_name in table.raw.keys():
                return table.raw.values()
        
        
    def _get_real_example_value(self, table_name: str, column_name: str) -> Any:
        try:
            query = f"SELECT {column_name} FROM {table_name} ORDER BY RAND() LIMIT 1"
            self.cursor.execute(query)
            result = self.cursor.fetchone()
            
            if result:
                value = result[column_name]
                
            if isinstance(value, decimal.Decimal):
                return float(value)
            elif isinstance(value, datetime.date):
                return value.strftime('%Y-%m-%d')
            elif isinstance(value, bytes):
                return value.decode('utf-8')
            else:
                return value
                
            return None
        except Exception as e:
            # 记录日志（可选）
            return None

    def _get_table_metadata(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        self.cursor.execute(f"""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, 
                   IS_NULLABLE, COLUMN_DEFAULT, {self.db_config.meta.column_comment_key}
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """, (self.db_config['database'],))
        columns = self.cursor.fetchall()

        # 获取外键约束
        self.cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, 
                   REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = %s
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """, (self.db_config['database'],))
        foreign_keys = defaultdict(list)
        for row in self.cursor.fetchall():
            foreign_keys[(row['TABLE_NAME'], row['COLUMN_NAME'])].append({
                'referenced_table': row['REFERENCED_TABLE_NAME'],
                'referenced_column': row['REFERENCED_COLUMN_NAME']
            })

        metadata = defaultdict(dict)
        for col in columns:
            table_name = f"{col['TABLE_NAME']} (describtion: {self._get_description(col['TABLE_NAME'])})"
            column_name = col['COLUMN_NAME']
            is_nullable = col['IS_NULLABLE']
            default = col['COLUMN_DEFAULT']
            column_comment = col['COLUMN_COMMENT']
            

            constraints = []
            if fk := foreign_keys.get((col['TABLE_NAME'], column_name)):
                for info in fk:
                    constraints.append(f"Foreign key to {info['referenced_table']}.{info['referenced_column']}")

            if is_nullable == 'NO':
                constraints.append('Not nullable')

            if default is not None:
                constraints.append(f'Default value: {default}')

            metadata[table_name][column_name] = {
                'Comment': column_comment,
            }

        return dict(metadata)