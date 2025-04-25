import ast
import json
import os
import re
from typing import Dict, Optional
from agent_engine.agent import ContextualChatEngine


class COCOClassCheckerAgent:
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        language: str = "english",
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512,
        context_length: int = 4096,
        vllm_cfg: Optional[dict] = None
    ):
        self.chat_engine = ContextualChatEngine(
            model_name=model_name,
            system_prompt=system_prompt,
            language = language,
            tmp_dir=tmp_dir,
            max_new_tokens=max_new_tokens,
            vllm_cfg=vllm_cfg
        )
        self.context_length = context_length

    def correct_coco_annotation(self, coco_info: Dict) -> Dict:
        image_path = coco_info.get("image_path")
        annotations = coco_info.get("annotations", [])
        allowed_classes = coco_info.get("allowed_classes", [])

        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"[bold red]Image file not found: [bold cyan]{image_path}[/bold cyan]")

        prompt = f"""
Original input:
The following is a COCO JSON data containing object detection results. Your output must strictly adhere to this structure:
{annotations}

Task:
Due to issues with the classification results, reassign the `category_id` based on the following allowed class space:
{allowed_classes}

Task requirements:
1. Only modify the `category_id` field for each object.
2. Keep the original COCO JSON structure and all other field values unchanged.
3. Do not provide any explanations; directly output the modified complete COCO JSON.
4. Use markdown syntax ```json```.

Abstract example:
- Original input:
[
    {{
        "id": m,
        "image_id": n,
        "category_id": u,
        "segmentation": [],
        "area": v,
        "bbox": [
            x,
            y,
            w,
            h
        ],
        "iscrowd": 0
    }}
]
- Output:
[
    {{
        "id": m,
        "image_id": n,
        "category_id": ${{fixed_category_id}},
        "segmentation": [],
        "area": v,
        "bbox": [
            x,
            y,
            w,
            h
        ],
        "iscrowd": 0
    }}
]"""

        response = self.chat_engine.generate_response(
            prompt, img_path=image_path)
        corrected_annotations = response["result"].strip()
        try:
            if "```json" in corrected_annotations:
                match = re.search(r"```json(.*?)```",
                                  corrected_annotations, re.DOTALL)
                corrected_annotations = match.group(1).strip() if match else ""
            corrected_annotations = json.loads(corrected_annotations)
        except json.JSONDecodeError:
            try:
                corrected_annotations = ast.literal_eval(corrected_annotations)
                corrected_annotations = json.dumps(corrected_annotations)
                corrected_annotations = json.loads(corrected_annotations)
            except (ValueError, SyntaxError) as e:
                self.chat_engine.console.print(f"[bold red]Invalid format: {corrected_annotations}. Error: [bold cyan]{e}[/bold cyan]")
                corrected_annotations = annotations

        id_list = [
            annotation["category_id"] for annotation in allowed_classes
        ]
        anno_id_list = [
            annotation["category_id"] for annotation in corrected_annotations
        ]
        for annotation in corrected_annotations:
            if any(
                cls not in id_list for cls in anno_id_list
            ):
                self.chat_engine.console.print(f"[bold red]Invalid category id generated: [bold cyan]{annotation['category_id']}[/bold cyan]")
                corrected_annotations = annotations
                break

        return corrected_annotations
