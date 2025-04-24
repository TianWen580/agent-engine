import json
from typing import Dict, Optional
from agent_engine.agent import ContextualChatEngine
from agent_engine.utils import BaikeWebCrawler

class BaikeSpeciesNameTranslateAgent:
    def __init__(
            self,
            model_name: str,
            system_prompt: str = "",
            language: str = "english",
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
            language=language,
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
            name_info = f"English name: {source_name}"
            target_lang = "Latin scientific name"
        elif mode == "la2en":
            name_info = f"Latin scientific name: {source_name}"
            target_lang = "English name"
            
        formatted_coco_category = json.dumps(coco_category, ensure_ascii=False, indent=4)
        
        return f"""
The original COCO-formatted category member is:
{formatted_coco_category}    

The name to be translated is: {name_info}

Wikipedia core text content (please carefully check for clues to translate into {target_lang}):
{wiki_content[:self.context]}...

Only replace the `name` attribute in the original COCO-formatted category. Do not add formatting like ```json```. For example:
{{
    "id": xx,
    "name": "new_name",
    "supercategory": "new_supercategory"
}}
""".strip()
