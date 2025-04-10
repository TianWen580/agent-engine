import json
from typing import Dict

import pandas as pd
from agent_engine.agent import ContextualChatEngine
from agent_engine.utils import BaikeWebCrawler

class BaikeResearchAgent:
    """Mixing web crawling and model analysis for species information."""
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
            context: int = 12000
        ):
        self.chat_engine = ContextualChatEngine(
            model_name=model_name,
            system_prompt=system_prompt,
            tmp_dir=tmp_dir,
            max_new_tokens=max_new_tokens
        )
        self.crawler = BaikeWebCrawler(
            storage_dir=storage_dir,
            storage_update_interval=storage_update_interval,
            secure_sleep_time=secure_sleep_time,
            sleep_time_variation=sleep_time_variation
        )
        self.context = context
    
    def query_species_info(self, chinese_name: str, latin_name: str) -> Dict:
        """Query species information based on the given names."""
        baidu_content = "（暂无）"
        wiki_content = "（暂无）"
        
        if chinese_name and not pd.isna(chinese_name):
            baidu_content = self.crawler.get_baidu_baike_content(chinese_name)
        elif latin_name and not pd.isna(latin_name):
            baidu_content = self.crawler.get_baidu_baike_content(latin_name)
        
        if latin_name:
            wiki_content = self.crawler.get_wikipedia_content(latin_name)
        
        prompt = self._build_prompt(chinese_name, latin_name, baidu_content, wiki_content)
        response = self.chat_engine.generate_response(prompt)
        return self._parse_response(response, f"{chinese_name}({latin_name})")

    def _build_prompt(self, chinese_name: str, latin_name: str, baidu_content: str, wiki_content: str) -> str:
        """Build a prompt for the chat engine."""
        name_info = f"中文名：{chinese_name}（拉丁学名：{latin_name}）"
        
        return f"""
请根据以下处理后的内容生成标准化报告：
{name_info}

百度百科核心文本内容：
{baidu_content[:self.context // 2]}...

维基百科核心文本内容（记录时请翻译成中文）：
{wiki_content[:self.context // 2]}...

需要输出的结构化信息：
1. 中国保护等级（仅限依据百度百科内容）
2. 国际濒危等级（依据所有百科内容）
3. 形态特征（详细描述和简要概括，依据所有百科，但是字数不要太多）
4. 生活习性（详细描述和简要概括，依据所有百科，但是字数不要太多）
5. 栖息环境（详细描述和简要概括，依据所有百科，但是字数不要太多）

请按以下JSON格式返回：
{{
    "中国保护等级": "...（在上下文寻找线索，只能用中文写一级/二级/三级/四级/非保护物种/待查，没找到写待查）",
    "国际濒危等级": "...（在上下文寻找线索，只能用中文写灭绝/野外灭绝/极危/濒危/易危/近危/无危/待查，没找到写待查）",
    "形态特征": {{ （在上下文寻找线索，用中文写，没找到写待查，不相关的内容不要放里面）
        "详细": "...",
        "简要": "..."
    }},
    "生活习性": {{ （在上下文寻找线索，用中文写，没找到写待查，不相关的内容不要放里面）
        "详细": "...",
        "简要": "..."
    }},
    "栖息环境": {{ （在上下文寻找线索，用中文写，没找到写待查，不相关的内容不要放里面）
        "详细": "...",
        "简要": "..."
    }}
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
                "中国保护等级": "[ERROR] Failed",
                "国际濒危等级": "[ERROR] Failed",
                "形态特征": {"详细": "[ERROR] Failed", "简要": "[ERROR] Failed"},
                "生活习性": {"详细": "[ERROR] Failed", "简要": "[ERROR] Failed"},
                "栖息环境": {"详细": "[ERROR] Failed", "简要": "[ERROR] Failed"}
            }