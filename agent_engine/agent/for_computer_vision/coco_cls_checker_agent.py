import ast
import json
import os
import re
from typing import Dict, Optional
from agent_engine.agent import ContextualChatEngine


class COCOClassCheckerAgent:
    """根据图片的COCO标注，纠正其物种分类结果"""

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512,
        context_length: int = 4096,
        vllm_cfg: Optional[dict] = None
    ):
        self.chat_engine = ContextualChatEngine(
            model_name=model_name,
            system_prompt=system_prompt,
            tmp_dir=tmp_dir,
            max_new_tokens=max_new_tokens,
            vllm_cfg=vllm_cfg
        )
        self.context_length = context_length

    def correct_coco_annotation(self, coco_info: Dict) -> Dict:
        """
        纠正COCO格式的目标检测标注。
        :param coco_annotation: 输入的COCO格式标注
        :return: 纠正后的COCO格式标注
        """
        image_path = coco_info.get("image_path")
        annotations = coco_info.get("annotations", [])
        allowed_classes = coco_info.get("allowed_classes", [])

        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"[AGENT] Image file not found: {image_path}")

        prompt = f"""
原始输入：
以下是一个包含目标检测结果的 coco JSON 数据，你的输出也要严格按照这个结构来：
{annotations}

任务：
由于分类结果存在问题，请根据以下允许的类别空间重新分配 `category_id`：
{allowed_classes}

任务要求：
1.仅修改每个目标的 `category_id` 字段
2.保持原始 coco JSON 的结构和所有其他字段的值不变
3.不要做任何解释，直接输出修改后的完整 coco JSON
4.使用markdown语法```json```

规范抽象示意：
- 原始输入：
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
- 输出：
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

        # 使用对话引擎生成响应
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
                print(f"[AGENT] Invalid format: {corrected_annotations}. Error: {e}")
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
                print(f"[AGENT] Invalid category id generated: {annotation['category_id']}")
                corrected_annotations = annotations
                break

        return corrected_annotations
