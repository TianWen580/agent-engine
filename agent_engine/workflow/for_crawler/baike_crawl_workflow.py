import os
import pandas as pd
from typing import Any, List, Dict
from agent_engine.workflow import BaseWorkflow


class BaikeSpeciesWorkflow(BaseWorkflow):
    def __init__(
            self,
            config: str,
    ):
        super().__init__(config)
        
        self.catalogue_paths = self.cfg.workflow.catalogue_paths
        self.catalogue_columns = self.cfg.workflow.catalogue_columns
        self.save_catalogue_columns = list(self.cfg.workflow.save.save_catalogue.raw.values())
        self.save_paths = self.cfg.workflow.save.save_paths
        self.save_columns = list(self.cfg.workflow.save.save_columns.raw.keys())
        self.result_columns = self.save_catalogue_columns.copy()
        self.result_columns.extend(self.save_columns)
        self.species_name_key = self.cfg.workflow.save.save_catalogue.name
        self.latin_name_key = self.cfg.workflow.save.save_catalogue.latin_name

    def _init_agents(self):
        self.agent = self.agent_class(
            model_name=self.cfg.workflow.agent.model_name,
            system_prompt=self.cfg.workflow.agent.system_prompt,
            language=self.cfg.workflow.agent.language,
            research_columns=self.cfg.workflow.save.save_columns.raw,
            storage_dir=self.cfg.workflow.storage.path,
            storage_update_interval=self.cfg.workflow.storage.update_interval,
            secure_sleep_time=self.cfg.workflow.secure_sleep.time,
            sleep_time_variation=self.cfg.workflow.secure_sleep.variation,
            tmp_dir=self.cfg.workflow.agent.tmp_dir,
            max_new_tokens=self.cfg.workflow.agent.max_new_tokens,
            context=self.cfg.workflow.agent.context,
            vllm_cfg=self.cfg.workflow.agent.vllm
        )

    def _save_excel(self, save_path, df: pd.DataFrame):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_excel(save_path, index=False)
        
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

            with self.bar(live_type="progress") as self.progress:
                task = self.progress.add_task(
                    "[cyan]Processing Species...", total=len(species_list))

                for count, species in enumerate(species_list, start=1):                    
                    species_name = species.get(self.species_name_key, "")
                    latin_name = species.get(self.latin_name_key, "")
                    
                    self.progress.update(task, description=f"[cyan]Processing {species_name}({latin_name})...")

                    try:
                        if not existing_df.empty:
                            matched_rows = existing_df[existing_df[self.latin_name_key]
                                                       == latin_name]

                            if not matched_rows.empty:
                                if not any("[WORKFLOW] Failed" in str(row.values) for _, row in matched_rows.iterrows()):
                                    self.progress.update(task, advance=1)
                                    print(f"[SKIP({count}/{len(species_list)})] {species_name}({latin_name}) already processed.")
                                    continue

                                existing_df = existing_df[~((existing_df[self.latin_name_key] == latin_name) &
                                                            (existing_df.apply(lambda row: "[WORKFLOW] Failed" in str(row.values), axis=1)))]
                                print(f"Removed failed records for {species_name}({latin_name}).")


                        species_info = self.agent.query_species_info(
                            species_name, latin_name)

                        formatted = {
                            self.species_name_key: species_name,
                            self.latin_name_key: latin_name,
                            **{key: species_info.get(key, "未找到") for key in self.result_columns if key not in [self.species_name_key, self.latin_name_key]}
                        }

                        current_result = pd.DataFrame([formatted])
                        existing_df = pd.concat(
                            [existing_df, current_result], ignore_index=True)

                        self._save_excel(save_path, existing_df)

                        self.agent.chat_engine.clear_context()
                    except Exception as e:
                        print(f"Failed: {e} for {species_name}({latin_name})")
                        error_result = {
                            col: "[WORKFLOW] Failed" for col in self.result_columns}
                        error_df = pd.DataFrame([error_result])

                        existing_df = pd.concat(
                            [existing_df, error_df], ignore_index=True)
                        self._save_excel(save_path, existing_df)

                    self.progress.update(task, advance=1)