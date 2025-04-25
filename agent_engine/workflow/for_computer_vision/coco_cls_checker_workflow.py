import json
import os
from typing import Dict, Any, List
import pandas as pd
from agent_engine.workflow import BaseWorkflow


class COCOClassCheckerWorkflow(BaseWorkflow):
    def __init__(self, config: str):
        super().__init__(config)
        
        self.coco_paths = self.cfg.workflow.input.coco_paths
        self.save_paths = self.cfg.workflow.save_paths

        if not isinstance(self.coco_paths, list):
            self.coco_paths = [
                os.path.join(self.cfg.workflow.input.coco_paths, file)
                for file in os.listdir(self.cfg.workflow.input.coco_paths)
                if file.endswith('.json')
            ]
        if not isinstance(self.save_paths, list):
            self.save_paths = [
                os.path.join(
                    self.cfg.workflow.save_paths, os.path.basename(coco_path))
                for coco_path in self.coco_paths
            ]

    def _init_agents(self):
        self.agent = self.agent_class(
            model_name=self.cfg.workflow.agent.model_name,
            system_prompt=self.cfg.workflow.agent.system_prompt,
            language=self.cfg.workflow.agent.language,
            tmp_dir=self.cfg.workflow.agent.tmp_dir,
            max_new_tokens=self.cfg.workflow.agent.max_new_tokens,
            vllm_cfg=self.cfg.workflow.agent.vllm,
        )

    def _save_results(self, save_path, corrected_annotations: List[Dict], coco_data: Dict):
        coco_data['annotations'] = corrected_annotations
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=4)
        print(f"Corrected annotations saved to {save_path}")
    
    def _execute(self):
        with self.bar(live_type="progress") as self.progress:
            outer_task = self.progress.add_task(
                "Processing COCO files...", total=len(self.coco_paths))

            for coco_path, save_path in zip(self.coco_paths, self.save_paths):
                self.progress.update(outer_task, description=f"[bold cyan]Processing COCO file: {coco_path}")
                coco_data = None
                corrected_annotations = []

                with open(coco_path, 'r') as f:
                    coco_data = json.load(f)

                allowed_classes = [
                    {"category_id": category['id'], "name": category['name']}
                    for category in coco_data['categories']
                    if category['name'] in self.cfg.workflow.allowed_classes
                ]

                inner_task = self.progress.add_task(
                    "[cyan]Processing images...", total=len(coco_data['images']))

                for image_info in coco_data['images']:
                    self.progress.update(inner_task, description=f"[cyan]Processing image: {image_info['file_name']}")
                    image_file_name = os.path.basename(image_info['file_name'])
                    annotations = [
                        ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
                    coco_info = {
                        "image_path": os.path.join(self.cfg.workflow.input.images_path, image_file_name),
                        "annotations": annotations,
                        "allowed_classes": allowed_classes
                    }

                    corrected_annotation_list = self.agent.correct_coco_annotation(
                        coco_info)
                    corrected_annotations.extend(corrected_annotation_list)
                    self.agent.chat_engine.clear_context()

                    self.progress.update(inner_task, advance=1)

                self._save_results(save_path, corrected_annotations, coco_data)

                self.progress.update(outer_task, advance=1)