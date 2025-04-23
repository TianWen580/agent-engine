import json
from typing import Dict, Optional
from agent_engine.agent import ContextualChatEngine
from agent_engine.utils import BaikeWebCrawler

class BaikeSpeciesNameTranslateAgent:
    def __init__(
            self,
            model_name: str,
            system_prompt: str = "",
            storage_dir: str = "storage",
            storage_update_interval: int = 30,
            secure_sleep_time: int = 2,
            sleep_time_variation: int = 1,
            tmp_dir: str = "asset/tmp",
            max_new_tokens: int = 512,
            context: int = 12000,
            vllm_cfg: Optional[dict] = None
        ):
        self.chat_engine = ContextualChatEngine(
            model_name=model_name,
            system_prompt=system_prompt,
            tmp_dir=tmp_dir,
            max_new_tokens=max_new_tokens,
            vllm_cfg=vllm_cfg
        )
        self.crawler = BaikeWebCrawler(
            storage_dir=storage_dir,
            storage_update_interval=storage_update_interval,
            secure_sleep_time=secure_sleep_time,
            sleep_time_variation=sleep_time_variation
        )
        self.context = context
        self.modes = ["en2la", "la2en"]
    
    def translate(self, coco_category, mode="en2la") -> Dict:
        """Query species information based on the given names."""
        if mode not in self.modes:
            assert f"Mode {mode} not supported."
            
        source_name = coco_category.get("name", "")
        
        wiki_content = ""
        
        wiki_content = self.crawler.get_wikipedia_content(source_name)
        
        prompt = self._build_prompt(coco_category, wiki_content, mode)
        response = self.chat_engine.generate_response(prompt)
        return response['result']

    def _build_prompt(self, coco_category: Dict, wiki_content: str, mode: str) -> str:
        """Build a prompt for the chat engine."""
        source_name = coco_category.get("name", "")

        if mode == "en2la":
            name_info = f"英文名：{source_name}"
            target_lang = "拉丁学名"
        elif mode == "la2en":
            name_info = f"拉丁学名：{source_name}"
            target_lang = "英文名"
            
        formatted_coco_category = json.dumps(coco_category, ensure_ascii=False, indent=4)
        
        return f"""
原始的 coco 格式的 category 成员是：
{formatted_coco_category}    

需要翻译的名字是：{name_info}

维基百科核心文本内容（清仔细检查是否有翻译成{target_lang}的线索）：
{wiki_content[:self.context]}...

只需要替换掉原始的 coco 格式的 category 的 name 属性，不要加```json```之类的格式，比如：
{{
    "id": xx,
    "name": "新的名字",
    "supercategory": "新的超类别"
}}
""".strip()