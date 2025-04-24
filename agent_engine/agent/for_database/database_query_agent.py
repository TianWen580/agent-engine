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
        
        db_config_raw = db_config.raw
        db_connection_config = {
            "host": db_config_raw["host"],
            "port": db_config_raw["port"],
            "user": db_config_raw["user"],
            "password": db_config_raw["password"],
            "database": db_config_raw["database"]
        }
        self.connection = mysql.connector.connect(**db_connection_config)
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
Based on the following database structure information:
{json.dumps(self.tables, indent=2, ensure_ascii=False)}

Translate the user's natural language query into a valid SQL statement:
User query: {user_query}

Requirements:
1. Use standard SQL syntax
2. Avoid special functions or stored procedures
3. Ensure correct table relationships
4. Make reasonable assumptions for potential ambiguities
5. You prefer to sort results by quantity in descending order
6. If the user's query is not in the database, return "(empty string)"

No explanations needed
        """

        response = self.chat_engine.generate_response(prompt)
        text = response['result'].strip()
        match = re.search(r"```sql(.*?)```", text, re.DOTALL)
        sql_command = match.group(1) if match else ""
        return sql_command

    def _generate_analysis(self, query: str, data: List[Dict]) -> str:
        """Generate an analysis report for the query results"""
        if not data:
            data = [{"result": "No query results"}]

        for row in data:
            for key, value in row.items():
                if isinstance(value, decimal.Decimal):
                    row[key] = float(value)
                if isinstance(value, bytes):
                    row[key] = value.decode('utf-8')
                # Convert datetime.date objects to strings
                if isinstance(value, datetime.date):
                    row[key] = value.strftime('%Y-%m-%d')

        is_too_long = len(data) > 20
        filter_result = json.dumps(data[:20], ensure_ascii=False, indent=2)

        prompt = f"""
Generate an analysis report based on the following query and results:
Original query: {query}
Total query results: {len(data)}:
{filter_result}
{'... (Too many results, only showing the first 20)' if is_too_long else ''}

Analysis requirements:
1. Summarize the main findings
2. Highlight key data points
3. If there are mathematical calculations, show the detailed process
4. If the query results are too many, be honest about your limitations
5. Keep it within 200 words
        """

        response = self.chat_engine.generate_response(prompt)
        analysis = "[Warning] Too many query results, unable to analyze in detail!\n" + \
            response['result'].strip(
            ) if is_too_long else response['result'].strip()
        return analysis
