import json
from typing import Dict, Optional
import pandas as pd
from agent_engine.agent import ContextualChatEngine
from agent_engine.utils import BaikeWebCrawler

class BaikeResearchAgent:
    def __init__(
            self,
            model_name: str,
            system_prompt: str = "",
            language: str = "english",
            research_columns: Dict[str, str] = None,
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
        self.language = language
        self.research_columns = research_columns
    
    def query_species_info(self, species_name: str, latin_name: str) -> Dict:
        baidu_content = "(404 not found)"
        wiki_content = "(404 not found)"
        
        if species_name and not pd.isna(species_name):
            baidu_content = self.crawler.get_baidu_baike_content(species_name)
        elif latin_name and not pd.isna(latin_name):
            baidu_content = self.crawler.get_baidu_baike_content(latin_name)
        
        if latin_name:
            wiki_content = self.crawler.get_wikipedia_content(latin_name)
        
        prompt = self._build_prompt(species_name, latin_name, baidu_content, wiki_content)
        response = self.chat_engine.generate_response(prompt)
        return self._parse_response(response, f"{species_name}({latin_name})")

    def _build_prompt(self, species_name: str, latin_name: str, baidu_content: str, wiki_content: str) -> str:
        """Build a prompt for the chat engine."""
        name_info = f"name: {species_name}(latin name: {latin_name})"

        research_querys = {
            **{key: f"... ({value}, please translate into {self.language})" for key, value in self.research_columns.items()}
        }

        return f"""
Please generate a standardized report based on the following processed content:
{name_info}

Baidu Baike core text content:
{baidu_content[:self.context // 2]}...

Wikipedia core text content (please translate into {self.language} when recording):
{wiki_content[:self.context // 2]}...

Please return the result in the following JSON format:
{{
    {research_querys}
}}""".strip()


    def _parse_response(self, response: dict, name: str) -> Dict:
        """Parse the response from the chat engine."""
        try:
            response_text = response.get('result', '')
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_content = response_text[start:end]
            return json.loads(json_content)
        except Exception as e:
            print(f"[ERROR] Json not formated well: {e} for {name}")
            return {
                **{key: f"[ERROR] Failed" for key in self.research_columns.keys()},
            }