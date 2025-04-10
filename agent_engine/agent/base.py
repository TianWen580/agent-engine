from abc import ABC, abstractmethod
import os
from typing import Dict, Optional

class BaseChatEngine(ABC):
    def __init__(
        self,
        model_name: str,
        max_pixels: int = 660 * 660,
        min_pixels: int = 128 * 128,
        system_prompt: Optional[str] = None,
        tmp_dir: str = "asset/tmp",
    ):
        self.model_name = model_name
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.system_prompt = system_prompt
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.context = []
        self.tasks: Dict[str, dict] = {}
        if system_prompt:
            self._init_system_prompt(system_prompt)

    @abstractmethod
    def _init_system_prompt(self, system_prompt: str):
        """Initialize the system prompt."""
        pass

    @abstractmethod
    def generate_response(self, prompt: str, img_path: Optional[str] = None) -> dict:
        """Generate a response based on the input prompt and optional image."""
        pass

    @abstractmethod
    def _process_task(self, job_id: str) -> dict:
        """Process a specific task with the given job ID."""
        pass

    @abstractmethod
    def clear_context(self):
        """Clear the context and clean up temporary files."""
        pass

    def __del__(self):
        self.clear_context()
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)