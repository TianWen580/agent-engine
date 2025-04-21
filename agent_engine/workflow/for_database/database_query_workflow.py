import json
import os
from typing import Dict, Any, List
import pandas as pd
from agent_engine.workflow import BaseWorkflow
from agent_engine.utils import import_class


class DatabaseQueryWorkflow(BaseWorkflow):
    def __init__(self, config: str):
        super().__init__(config)
        
        self.result_columns = [
            "query", "sql", "result",
            "analysis", "timestamp", "visualizations"
        ]
        self.results = []
        self.queries = self.cfg.workflow.queries

    def _init_agents(self):
        self.agent = [
            self.agent_class[0](
                model_name=self.cfg.workflow.agent.members[0].model_name,
                db_config=self.cfg.database,
                system_prompt=self.cfg.workflow.agent.members[0].system_prompt,
                tmp_dir=self.cfg.workflow.agent.members[0].tmp_dir,
                max_new_tokens=self.cfg.workflow.agent.members[0].max_new_tokens
            ),
            self.agent_class[1](
                model_name=self.cfg.workflow.agent.members[1].model_name,
                system_prompt=self.cfg.workflow.agent.members[1].system_prompt,
                tmp_dir=self.cfg.workflow.agent.members[1].tmp_dir,
                max_new_tokens=self.cfg.workflow.agent.members[1].max_new_tokens
            )
        ]

    def _save_results(self, results: List[Dict]):
        df = pd.DataFrame(results)
        file_path = self.cfg.workflow.save_path
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
        with self._live_display(live_type="progress") as progress:
            task = progress.add_task("[cyan]处理查询...", total=len(self.queries))

            for query in self.queries:
                progress.update(task, description=f"[cyan]处理查询: {query}")

                # 数据库查询阶段
                db_result = self.agent[0].natural_query(query)
                self.results.append({
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
                #     visual_paths = self.agent[1].generate_visuals(
                #         user_query=query,
                #         data_structure=data_structure,
                #         data_sample=db_result['result']
                #     )
                #     results[-1]['visualizations'] = json.dumps(visual_paths)

                if not self.cfg.workflow.agent.members[0].contextualize:
                    self.agent[0].chat_engine.clear_context()
                self.agent[1].chat_engine.clear_context()
                self._save_results(self.results)
                progress.console.print(f"[green][WORKFLOW] 结果已更新至 {self.cfg.workflow.save_path}")

                progress.advance(task)

    def _post_execute(self):
        if self.cfg.workflow.verbose:
            self._verbose_output(self.cfg.workflow.save_path)
