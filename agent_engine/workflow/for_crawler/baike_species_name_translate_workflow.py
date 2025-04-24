import json
import os
import pandas as pd
from typing import Any, List, Dict
from agent_engine.workflow import BaseWorkflow


class BaikeSpeciesNameTranslateWorkflow(BaseWorkflow):
    def __init__(
            self,
            config: str,
    ):
        super().__init__(config)

    def _init_agents(self):
        self.agent = self.agent_class(
            model_name=self.cfg.workflow.agent.model_name,
            system_prompt=self.cfg.workflow.agent.system_prompt,
            language=self.cfg.workflow.agent.language,
            storage_dir=self.cfg.workflow.storage.path,
            storage_update_interval=self.cfg.workflow.storage.update_interval,
            secure_sleep_time=self.cfg.workflow.secure_sleep.time,
            sleep_time_variation=self.cfg.workflow.secure_sleep.variation,
            tmp_dir=self.cfg.workflow.agent.tmp_dir,
            max_new_tokens=self.cfg.workflow.agent.max_new_tokens,
            context=self.cfg.workflow.agent.context,
            vllm_cfg=self.cfg.workflow.agent.vllm
        )

    def _execute(self, coco_file: str) -> Dict:
        with open(coco_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        if 'categories' not in coco_data:
            raise ValueError("[WORKFLOW] The provided COCO file does not contain a 'categories' key.")

        translated_categories = []
        for coco_category in coco_data['categories']:
            new_coco_category = self.agent.translate(
                coco_category,
                mode=self.cfg.workflow.agent.mode
            )
            translated_categories.append(new_coco_category)
            self.agent.chat_engine.clear_context()

        coco_data['categories'] = translated_categories
        save_path = self._save_coco(coco_data, coco_file)

        return {"saved_coco_file": save_path}

    def _save_coco(self, coco_data: Dict, original_coco_file: str) -> str:
        save_dir = self.cfg.workflow.save_path
        os.makedirs(save_dir, exist_ok=True)

        original_file_name = os.path.basename(original_coco_file)
        file_name, file_ext = os.path.splitext(original_file_name)
        new_file_name = f"{file_name}_translated{file_ext}"

        save_path = os.path.join(save_dir, new_file_name)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=4)

        return save_path