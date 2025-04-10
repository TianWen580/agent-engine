import json
import os
from typing import Dict, Any, List
import pandas as pd
from agent_engine.workflow import BaseWorkflow
from agent_engine.utils import import_class

class DatabaseQueryWorkflow(BaseWorkflow):
    def __init__(self, config: str):
        super().__init__(config)
        self._init_agents()
        self.result_columns = [
            "query", "sql", "result", 
            "analysis", "timestamp", "visualizations"
        ]

    def _init_agents(self):
        db_agent_class = import_class(self.cfg['workflow']['agent']['type'])
        self.db_agent = db_agent_class(
            model_name=self.cfg['workflow']['agent']['model_name'],
            db_config=self.cfg['database'],
            system_prompt=self.cfg['workflow']['agent']['system_prompt'],
            tmp_dir=self.cfg['workflow']['agent']['tmp_dir'],
            max_new_tokens=self.cfg['workflow']['agent']['max_new_tokens']
        )
        
        visual_agent_class = import_class(self.cfg['workflow']['visual_agent']['type'])
        self.visual_agent = visual_agent_class(
            model_name=self.cfg['workflow']['visual_agent']['model_name'],
            system_prompt=self.cfg['workflow']['visual_agent']['system_prompt'],
            tmp_dir=self.cfg['workflow']['visual_agent']['tmp_dir'],
            max_new_tokens=self.cfg['workflow']['visual_agent']['max_new_tokens']
        )

    def _save_results(self, results: List[Dict]):
        df = pd.DataFrame(results)
        file_path = self.cfg['workflow']['save_path']
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_excel(file_path, index=False)

    def _verbose_output(self, save_path: str):
        df = pd.read_excel(save_path)
        print("-" * 50)
        print("[VERBOSE OUTPUT]\n")
        for _, row in df.iterrows():
            print(f"[查询语句]\n\n{row['query']}\n")
            print(f"[生成SQL]\n\n{row['sql']}\n")
            print(f"[分析结果]\n\n{row['analysis']}\n")
            print("-" * 50)

    def _execute(self):
        results = []
        queries = self.cfg['workflow']['queries']

        with self._live_display(live_type="progress") as progress:
            task = progress.add_task("[cyan]处理查询...", total=len(queries))

            for query in queries:
                progress.update(task, description=f"[cyan]处理查询: {query}")
                
                # 数据库查询阶段
                db_result = self.db_agent.natural_query(query)
                results.append({
                    "query": query,
                    "sql": db_result.get('sql', ''),
                    "result": json.dumps(db_result.get('result', []), ensure_ascii=False),
                    "analysis": db_result.get('analysis', ''),
                    "timestamp": pd.Timestamp.now(),
                    "visualizations": "[]"
                })
                
                # 可视化生成阶段（注释部分）
                # if db_result['status'] == 'success' and db_result['result']:
                #     data_structure = {k: type(v).__name__ for k, v in db_result['result'][0].items()}
                #     visual_paths = self.visual_agent.generate_visuals(
                #         user_query=query,
                #         data_structure=data_structure,
                #         data_sample=db_result['result']
                #     )
                #     results[-1]['visualizations'] = json.dumps(visual_paths)
                
                if not self.cfg['workflow']['agent']['contextualize']:
                    self.db_agent.chat_engine.clear_context()
                self.visual_agent.chat_engine.clear_context()
                self._save_results(results)
                progress.console.print(f"[green][WORKFLOW] 结果已更新至 {self.cfg['workflow']['save_path']}")

                progress.advance(task)

        if self.cfg['workflow']['verbose']:
            self._verbose_output(self.cfg['workflow']['save_path'])