from abc import ABC, abstractmethod
import os
import torch
from typing import Dict, Optional, Union, Type
from transformers import AutoModelForCausalLM, AutoProcessor
from dataclasses import dataclass
import os
from rich.console import Console
from rich import print
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize rich console
console = Console()

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False

@dataclass
class ModelConfig:
    model_class: Type
    processor_class: Type = AutoProcessor
    supports_images: bool = False
    supports_vllm: bool = False

class BaseChatEngine(ABC):
    MODEL_CONFIGS = {
        'Qwen2.5-VL': ModelConfig(
            model_class=Qwen2_5_VLForConditionalGeneration if HAS_QWEN_VL else None,
            processor_class=AutoProcessor,
            supports_images=True,
            supports_vllm=False
        )
    }
    DEFAULT_CONFIG = ModelConfig(
        model_class=AutoModelForCausalLM,
        processor_class=AutoProcessor,
        supports_images=False,
        supports_vllm=True
    )

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512,
        vllm_cfg: Optional[dict] = None,
    ):
        self.is_online = model_name.startswith('http')
        self.model_config = self._get_model_config(model_name)
        self.tmp_dir = tmp_dir
        self.max_new_tokens = max_new_tokens
        self.vllm_cfg = vllm_cfg
        self.tasks = {}
        self.context = []
        self.system_prompt = None
        
        if not self.is_online and self.model_config.model_class is None:
            raise ValueError(f"Model {model_name} is not properly supported")

        if vllm_cfg.enable is None:
            self.use_vllm = False
            console.print("[bold yellow][AGENT][/bold yellow] VLLM usage not specified, defaulting to False")
        elif vllm_cfg.enable and not self.is_online:
            try:
                import vllm
                VLLM_AVAILABLE = True
            except ImportError:
                VLLM_AVAILABLE = False
            self.use_vllm = (VLLM_AVAILABLE and not self.is_online 
                           and self.model_config.supports_vllm
                           and self._check_gpu_compatibility())
            if self.use_vllm:
                console.print("[bold green][AGENT][/bold green] VLLM enabled for acceleration")
            else:
                console.print("[bold yellow][AGENT][/bold yellow] VLLM not available or not supported for this model")
        elif not self.is_online:
            self.use_vllm = False
            console.print("[bold yellow][AGENT][/bold yellow] VLLM disabled")
            
        if self.is_online:
            parts = model_name.split('@', 2)
            if len(parts) != 3:
                raise ValueError("[bold red][AGENT][/bold red] Invalid model name for online mode.")
            self.api_url, self.api_key, self.api_model = parts
            console.print(f"[bold blue][AGENT][/bold blue] Initializing online model with API URL: {self.api_url} for {self.api_model}")
            self.model = None
            self.processor = None
        else:
            if self.use_vllm:
                self._init_vllm_model(model_name)
            else:
                self._init_transformers_model(model_name)

        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        
        if system_prompt:
            self._init_system_prompt(system_prompt)

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Determine the appropriate model configuration."""
        if self.is_online:
            return self.DEFAULT_CONFIG
            
        for pattern, config in self.MODEL_CONFIGS.items():
            if pattern in model_name:
                console.print(f"[bold blue][AGENT][/bold blue] Using model configuration {pattern} for {model_name}")
                return config
                
        console.print(f"[bold yellow][AGENT] Using default model configuration")
        return self.DEFAULT_CONFIG

    def _check_gpu_compatibility(self) -> bool:
        """Check if GPU supports required features for VLLM."""
        if not torch.cuda.is_available():
            return False
            
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            console.print(f"[bold yellow][AGENT][/bold yellow] GPU compute capability {major}.{minor} may have limited VLLM support")
        return True

    def _init_vllm_model(self, model_name: str):
        """Initialize model using VLLM."""
        from vllm import LLM, SamplingParams
        
        console.print("[bold blue][AGENT][/bold blue] Attempting to use VLLM for acceleration")
        try:
            dtype = 'float16' if torch.cuda.get_device_capability()[0] < 8 else 'auto'
            self.model = LLM(
                model=model_name,
                tensor_parallel_size=self.vllm_cfg.tensor_parallel_size,
                max_num_seqs=self.max_new_tokens,
                max_model_len=self.vllm_cfg.max_model_len,
                dtype=dtype,
                trust_remote_code=True,
                gpu_memory_utilization=self.vllm_cfg.gpu_memory_utilization,
            )
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=self.max_new_tokens
            )
            self.processor = None
            console.print("[bold green][AGENT][/bold green] VLLM initialization successful")
        except Exception as e:
            console.print(f"[bold red][AGENT][/bold red] VLLM initialization failed: {str(e)}, rolling back to transformers")
            self.model = None
            self.processor = None
            self._init_transformers_model(model_name)
            self.use_vllm = False

    def _init_transformers_model(self, model_name: str):
        """Initialize model using transformers library."""
        console.print(f"[bold blue][AGENT][/bold blue] Implementing transformers model: {model_name}")
        try:
            if torch.cuda.is_available():
                from flash_attn import FlashMHA
                from flash_attn.modules.mha import FlashMHA as FlashAttention
                from flash_attn.modules.mha import FlashMHA as FlashAttention2
                console.print("[bold green][AGENT][/bold green] Using flash attention for acceleration")
            attn_implementation = "flash_attention_2"
        except ImportError:
            console.print("[bold yellow][AGENT][/bold yellow] Flash attention not available, using eager")
            attn_implementation = "eager"
        self.model = self.model_config.model_class.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = self.model_config.processor_class.from_pretrained(
            model_name, 
            use_fast=True
        ) if self.model_config.processor_class else None

    def _init_system_prompt(self, system_prompt: str):
        """Initialize the system prompt."""
        if self.model_config.supports_images:
            sys = [{"type": "text", "text": system_prompt}]
        else:
            sys = system_prompt
        self.system_prompt = {
            "role": "system",
            "content": sys
        }
        self.context.append(self.system_prompt)

    def clear_context(self):
        """Clear the context and clean up temporary files."""
        if hasattr(self, 'context'):
            self.context = [self.system_prompt] if hasattr(self, 'system_prompt') and self.system_prompt else []
        
        if hasattr(self, 'tasks'):
            for task in self.tasks.values():
                if task.get('image_path'):
                    try:
                        os.remove(task['image_path'])
                    except FileNotFoundError:
                        pass
            self.tasks = {}
            
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            for file in os.listdir(self.tmp_dir):
                try:
                    os.remove(os.path.join(self.tmp_dir, file))
                except Exception as e:
                    console.print(f"[bold yellow][AGENT] Failed to delete temporary file {file}: {str(e)}")

    def __del__(self):
        try:
            self.clear_context()
        except Exception as e:
            console.print(f"[bold yellow][AGENT] Error during cleanup: {str(e)}")

    @abstractmethod
    def generate_response(self, prompt: str, img_path: Optional[str] = None) -> dict:
        pass

    @abstractmethod
    def _process_task(self, job_id: str) -> dict:
        pass
