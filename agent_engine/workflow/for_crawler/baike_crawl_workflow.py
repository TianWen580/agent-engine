import os
import pandas as pd
from typing import Any, List, Dict
from agent_engine.workflow import BaseWorkflow
from agent_engine.utils import import_class


class BaikeSpeciesWorkflow(BaseWorkflow):
    def __init__(
            self,
            config: str,
    ):
        super().__init__(config)
        self._init_agent()

        self.result_columns = [
            "中文名", "拉丁名",
            "中国保护等级", "国际濒危等级",
            "形态特征(详细)", "形态特征(简化)",
            "生活习性(详细)", "生活习性(简化)",
            "栖息环境(详细)", "栖息环境(简化)"
        ]

    def _init_agent(self):
        agent_class = import_class(self.cfg.workflow.agent.type)
        self.agent = agent_class(
            model_name=self.cfg.workflow.agent.model_name,
            system_prompt=self.cfg.workflow.agent.system_prompt,
            storage_dir=self.cfg.workflow.storage.path,
            storage_update_interval=self.cfg.workflow.storage.update_interval,
            secure_sleep_time=self.cfg.workflow.secure_sleep.time,
            sleep_time_variation=self.cfg.workflow.secure_sleep.variation,
            tmp_dir=self.cfg.workflow.agent.tmp_dir,
            max_new_tokens=self.cfg.workflow.agent.max_new_tokens,
            context=self.cfg.workflow.agent.context
        )

    def _save_excel(self, save_path, df: pd.DataFrame):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_excel(save_path, index=False)
        
    def _pre_execute(self):
        self.catalogue_paths = self.cfg.workflow.catalogue_paths
        self.catalogue_columns = self.cfg.workflow.catalogue_columns
        self.save_catalogue_columns = self.cfg.workflow.save.save_catalogue_columns
        self.save_paths = self.cfg.workflow.save.save_paths

    def _execute(self) -> pd.DataFrame:
        for path, save_path in zip(self.catalogue_paths, self.save_paths):
            species_list = pd.read_excel(path, usecols=self.catalogue_columns)
            self.save_catalogue_columns.reverse()
            species_list.columns = self.save_catalogue_columns
            species_list = species_list.to_dict(orient="records")

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            if os.path.exists(save_path):
                existing_df = pd.read_excel(save_path)
            else:
                existing_df = pd.DataFrame(columns=self.result_columns)

            with self._live_display(live_type="progress") as progress:
                task = progress.add_task(
                    "[cyan]Processing Species...", total=len(species_list))

                for count, species in enumerate(species_list, start=1):
                    chinese_name = species.get("中文名", "")
                    latin_name = species.get("拉丁名", "")

                    try:
                        if not existing_df.empty:
                            matched_rows = existing_df[existing_df["拉丁名"]
                                                       == latin_name]

                            if not matched_rows.empty:
                                if not any("[WORKFLOW] Failed" in str(row.values) for _, row in matched_rows.iterrows()):
                                    progress.update(task, advance=1)
                                    print(
                                        f"[WORKFLOW][SKIP({count}/{len(species_list)})] {chinese_name}({latin_name}) already processed.")
                                    continue

                                existing_df = existing_df[~((existing_df["拉丁名"] == latin_name) &
                                                            (existing_df.apply(lambda row: "[WORKFLOW] Failed" in str(row.values), axis=1)))]
                                print(
                                    f"[WORKFLOW] Removed failed records for {chinese_name}({latin_name}).")

                        progress.update(
                            task, description=f"[cyan]Processing {chinese_name}({latin_name})...")
                        species_info = self.agent.query_species_info(
                            chinese_name, latin_name)

                        formatted = {
                            "中文名": chinese_name,
                            "拉丁名": latin_name,
                            "中国保护等级": species_info.get("中国保护等级", "未找到"),
                            "国际濒危等级": species_info.get("国际濒危等级", "未找到"),
                            "形态特征(详细)": species_info["形态特征"]["详细"],
                            "形态特征(简化)": species_info["形态特征"]["简要"],
                            "生活习性(详细)": species_info["生活习性"]["详细"],
                            "生活习性(简化)": species_info["生活习性"]["简要"],
                            "栖息环境(详细)": species_info["栖息环境"]["详细"],
                            "栖息环境(简化)": species_info["栖息环境"]["简要"]
                        }

                        current_result = pd.DataFrame([formatted])
                        existing_df = pd.concat(
                            [existing_df, current_result], ignore_index=True)

                        self._save_excel(save_path, existing_df)

                        self.agent.chat_engine.clear_context()
                    except Exception as e:
                        print(
                            f"[WORKFLOW] Failed: {e} for {chinese_name}({latin_name})")
                        error_result = {
                            col: "[WORKFLOW] Failed" for col in self.result_columns}
                        error_df = pd.DataFrame([error_result])

                        existing_df = pd.concat(
                            [existing_df, error_df], ignore_index=True)
                        self._save_excel(save_path, existing_df)

                    progress.update(task, advance=1)