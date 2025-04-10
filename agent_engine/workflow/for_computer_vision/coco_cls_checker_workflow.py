import json
import os
from typing import Dict, Any, List
import pandas as pd
from agent_engine.workflow import BaseWorkflow
from agent_engine.utils import import_class
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn


class COCOClassCheckerWorkflow(BaseWorkflow):
    def __init__(self, config: str):
        super().__init__(config)
        self._init_agents()

    def _init_agents(self):
        agent_class = import_class(self.cfg['workflow']['agent']['type'])
        self.agent = agent_class(
            model_name=self.cfg['workflow']['agent']['model_name'],
            system_prompt=self.cfg['workflow']['agent']['system_prompt'],
            tmp_dir=self.cfg['workflow']['agent']['tmp_dir'],
            max_new_tokens=self.cfg['workflow']['agent']['max_new_tokens']
        )

    def _save_results(self, save_path, corrected_annotations: List[Dict], coco_data: Dict):
        coco_data['annotations'] = corrected_annotations
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=4)
        print(f"[WORKFLOW] Corrected annotations saved to {save_path}")

    def _execute(self):
        coco_paths = self.cfg['workflow']['input']['coco_paths']
        save_paths = self.cfg['workflow']['save_paths']

        if not isinstance(coco_paths, list):
            coco_paths = [
                os.path.join(self.cfg['workflow']['input']['coco_paths'], file)
                for file in os.listdir(self.cfg['workflow']['input']['coco_paths'])
                if file.endswith('.json')
            ]
        if not isinstance(save_paths, list):
            save_paths = [
                os.path.join(
                    self.cfg['workflow']['save_paths'], os.path.basename(coco_path))
                for coco_path in coco_paths
            ]

        with self._live_display(live_type="progress") as progress:
            outer_task = progress.add_task(
                "[bold green]Processing COCO files...", total=len(coco_paths))

            for coco_path, save_path in zip(coco_paths, save_paths):
                progress.console.print(
                    f"[WORKFLOW] Processing COCO file: {coco_path}")
                coco_data = None
                corrected_annotations = []

                with open(coco_path, 'r') as f:
                    coco_data = json.load(f)

                allowed_classes = [
                    {"category_id": category['id'], "name": category['name']}
                    for category in coco_data['categories']
                    if category['name'] in self.cfg['workflow']['allowed_classes']
                ]

                inner_task = progress.add_task(
                    "[cyan]Processing images...", total=len(coco_data['images']))

                for image_info in coco_data['images']:
                    image_file_name = os.path.basename(image_info['file_name'])
                    annotations = [
                        ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
                    coco_info = {
                        "image_path": os.path.join(self.cfg['workflow']['input']['images_path'], image_file_name),
                        "annotations": annotations,
                        "allowed_classes": allowed_classes
                    }

                    corrected_annotation_list = self.agent.correct_coco_annotation(
                        coco_info)
                    corrected_annotations.extend(corrected_annotation_list)
                    self.agent.chat_engine.clear_context()

                    progress.update(inner_task, advance=1)

                self._save_results(save_path, corrected_annotations, coco_data)

                progress.update(outer_task, advance=1)
