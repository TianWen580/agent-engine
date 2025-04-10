import datetime
import decimal
import re
from collections import defaultdict
from typing import Dict, Any

class DatabaseMetadata:
    def __init__(self, cursor, db_config):
        self.cursor = cursor
        self.db_config = db_config

    def _get_description(self, table_name: str) -> str:
        """获取表的描述信息"""
        descriptions = {
            "ob_detailview": "调查物种明细表，包含样线、样点、样方、随机调查",
            "ob_slsummary": "仅包含部分样线的调查记录表，仅记录部分样线调查的基本信息",
            "ob_speciesinfo": "物种底库表，记录全国所有物种的各种基本信息"
        }
        return descriptions.get(table_name, "")
        
    def _get_real_example_value(self, table_name: str, column_name: str) -> Any:
        """从真实数据中获取示例值"""
        try:
            # 使用随机排序获取一个真实值
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
        """获取包含示例和约束的数据库元数据"""
        # 获取列基本信息
        self.cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, 
                   IS_NULLABLE, COLUMN_DEFAULT, COLUMN_COMMENT
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

        # 构建元数据
        metadata = defaultdict(dict)
        for col in columns:
            table_name = f"{col['TABLE_NAME']}（描述：{self._get_description(col['TABLE_NAME'])}）"
            column_name = col['COLUMN_NAME']
            data_type = col['DATA_TYPE']
            column_type = col['COLUMN_TYPE']
            is_nullable = col['IS_NULLABLE']
            default = col['COLUMN_DEFAULT']
            column_comment = col['COLUMN_COMMENT']
            

            # 获取真实示例值
            real_example = self._get_real_example_value(col['TABLE_NAME'], column_name)
            
            # 如果获取不到真实值，使用类型生成示例值（回退机制）
            example = real_example if real_example is not None else f"<{data_type}>"

            # 处理外键约束
            constraints = []
            if fk := foreign_keys.get((col['TABLE_NAME'], column_name)):
                for info in fk:
                    constraints.append(f"外键到 {info['referenced_table']}.{info['referenced_column']}")

            # 处理非空约束
            if is_nullable == 'NO':
                constraints.append('非空')

            # 处理默认值
            if default is not None:
                constraints.append(f'默认值: {default}')

            # 组装列信息
            metadata[table_name][column_name] = {
                # '数据类型': data_type,
                # '示例值': example,
                '注释': column_comment,
            }

        return dict(metadata)