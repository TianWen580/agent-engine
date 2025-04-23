import datetime
import decimal
import json
import re
import mysql.connector
from typing import Dict, List, Optional
from agent_engine.agent import ContextualChatEngine
from agent_engine.utils import DatabaseMetadata


class DatabaseQueryAgent:
    def __init__(
        self,
        model_name: str,
        db_config: Dict,
        system_prompt: str = "",
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512,
        context_length: int = 4096,
        vllm_cfg: Optional[dict] = None
    ):
        self.db_config = db_config
        self.connection = mysql.connector.connect(**db_config.raw)
        self.cursor = self.connection.cursor(dictionary=True)

        self.tables = self._get_table_metadata()

        self.chat_engine = ContextualChatEngine(
            model_name=model_name,
            system_prompt=system_prompt,
            tmp_dir=tmp_dir,
            max_new_tokens=max_new_tokens,
            vllm_cfg=vllm_cfg
        )
        self.context_length = context_length

    def _get_table_metadata(self) -> Dict[str, Dict[str, str]]:
        """获取数据库表结构元数据，优化为列名到数据类型的映射"""
        metadata = DatabaseMetadata(self.cursor, self.db_config)
        return metadata._get_table_metadata()

    def natural_query(self, user_query: str) -> Dict:
        sql = ""
        analysis = ""

        try:
            sql = self._generate_sql(user_query)

            if sql != "":
                self.cursor.execute(sql)
                result = self.cursor.fetchall()
            else:
                result = [{"result": "数据不足，无法查询"}]

            analysis = self._generate_analysis(user_query, result)

            return {
                "status": "success",
                "query": user_query,
                "sql": sql,
                "result": result,
                "analysis": analysis
            }

        except Exception as e:
            return {
                "status": "error",
                "query": user_query,
                "sql": sql,
                "result": str(e),
                "analysis": analysis
            }

    def _generate_sql(self, user_query: str) -> str:
        prompt = f"""
根据以下数据库结构信息：
{json.dumps(self.tables, indent=2, ensure_ascii=False)}

将用户的自然语言查询转换为有效的SQL语句：
用户查询：{user_query}

要求：
1. 使用标准SQL语法
2. 避免使用特殊函数或存储过程
3. 确保表关联关系正确
4. 对可能的歧义进行合理假设
5. 你很喜欢按数量对结果进行降序排序
6. 如果用户想查询的内容不在数据库中，请返回“(空字符串)”

不要任何解释
        """

        response = self.chat_engine.generate_response(prompt)
        text = response['result'].strip()
        match = re.search(r"```sql(.*?)```", text, re.DOTALL)
        sql_command = match.group(1) if match else ""
        return sql_command

    def _generate_analysis(self, query: str, data: List[Dict]) -> str:
        """生成查询结果分析报告"""
        if not data:
            data = [{"result": "查询结果为空"}]

        for row in data:
            for key, value in row.items():
                if isinstance(value, decimal.Decimal):
                    row[key] = float(value)
                if isinstance(value, bytes):
                    row[key] = value.decode('utf-8')
                # 把datetime.date对象转换为字符串
                if isinstance(value, datetime.date):
                    row[key] = value.strftime('%Y-%m-%d')

        is_too_long = len(data) > 20
        filter_result = json.dumps(data[:20], ensure_ascii=False, indent=2)

        prompt = f"""
根据以下查询和结果生成分析报告：
原始查询：{query}
查询结果共{len(data)}条：
{filter_result}
{'...（查询结果太多，只展示前20条）' if is_too_long else ''}

分析要求：
1. 总结主要发现
2. 指出关键数据点
3. 如果有数学计算，请详细展示计算过程
4. 如果查询结果太多，请不要信口开河，诚实告知你的局限性
5. 控制在200字以内
        """

        response = self.chat_engine.generate_response(prompt)
        analysis = "[警告] 查询结果太多，无法进行详细分析！\n" + \
            response['result'].strip(
            ) if is_too_long else response['result'].strip()
        return analysis
