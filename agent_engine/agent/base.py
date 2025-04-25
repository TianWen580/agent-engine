from abc import ABC, abstractmethod
import os
import torch
from typing import Dict, Optional, Union, Type
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoProcessor
from dataclasses import dataclass
from agent_engine.utils.warpper import VerboseConsoleWrapper

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HAS_QWEN_VL = True
except ImportError:
    HAS_QWEN_VL = False

@dataclass
class ModelConfig:
    """_summary_

    Args:
        model_class (Type): The transformers class of the model to be used.
        processor_class (Type): The transformers class of the processor to be used.
        supports_images (bool): Whether the model supports image inputs.
        supports_vllm (bool): Whether the model supports VLLM acceleration.
    """
    model_class: Type
    processor_class: Type = AutoProcessor
    supports_images: bool = False
    supports_vllm: bool = False
    
class BaseChatEngine(ABC):
    """_summary_

    Args:
        model_name (str): The name of the model to be used.
        system_prompt (str): The system prompt to be used.
        language (str): The language for the model response.
        tmp_dir (str): The directory for temporary files.
        max_new_tokens (int): The maximum number of new tokens to generate.
        vllm_cfg (Optional[dict]): Configuration for VLLM acceleration.

    Raises:
        ValueError: If the model is not properly supported.
        ImportError: If VLLM is not available and the model requires it.
        RuntimeError: If the model initialization fails.
        Exception: If any other error occurs during initialization.
    
    This class serves as a base class for chat engines that interact with various models.
    It provides methods for initializing the model, processing tasks, and generating responses.
    The class is designed to be extended by specific implementations for different models.
    """
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
        language: str = "english",
        tmp_dir: str = "asset/tmp",
        max_new_tokens: int = 512,
        vllm_cfg: Optional[dict] = None,
    ):
        self.console = VerboseConsoleWrapper(Console(), role="AGENT")
        self.is_online = model_name.startswith('http')
        self.model_config = self._get_model_config(model_name)
        self.tmp_dir = tmp_dir
        self.max_new_tokens = max_new_tokens
        self.vllm_cfg = vllm_cfg
        self.tasks = {}
        self.context = []
        self.language = language
        self.system_prompt = system_prompt
                
        if not self.is_online and self.model_config.model_class is None:
            raise ValueError(f"Model {model_name} is not properly supported")

        if vllm_cfg.enable is None:
            self.use_vllm = False
            self.console.print("[bold yellow]VLLM usage not specified, defaulting to False")
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
                self.console.print("[bold green]VLLM enabled for acceleration")
            else:
                self.console.print("[bold yellow]VLLM not available or not supported for this model")
        elif not self.is_online:
            self.use_vllm = False
            self.console.print("[bold yellow]VLLM disabled")
            
        if self.is_online:
            parts = model_name.split('@', 2)
            if len(parts) != 3:
                raise ValueError("Invalid model name for online mode.")
            self.api_url, self.api_key, self.api_model = parts
            self.console.print(f"Initializing online model with API URL: [bold cyan]{self.api_url}[/bold cyan] for [bold cyan]{self.api_model}[/bold cyan]")
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
            self._init_system_prompt()

    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Determine the appropriate model configuration."""
        if self.is_online:
            return self.DEFAULT_CONFIG
            
        for pattern, config in self.MODEL_CONFIGS.items():
            if pattern in model_name:
                self.console.print(f"Using model configuration [bold cyan]{pattern}[/bold cyan] for [bold cyan]{model_name}[/bold cyan]")
                return config
                
        self.console.print(f"[bold yellow]Using default model configuration")
        return self.DEFAULT_CONFIG

    def _check_gpu_compatibility(self) -> bool:
        """Check if GPU supports required features for VLLM."""
        if not torch.cuda.is_available():
            return False
            
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            self.console.print(f"[bold yellow]GPU compute capability [bold cyan]{major}.{minor}[/bold cyan] may have limited VLLM support")
        return True

    def _init_vllm_model(self, model_name: str):
        """Initialize model using VLLM."""
        from vllm import LLM, SamplingParams
        
        self.console.print("Attempting to use VLLM for acceleration")
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
            self.console.print("[bold green]VLLM initialization successful")
        except Exception as e:
            self.console.print(f"VLLM initialization failed: [bold red]{str(e)}[/bold red]")
            self.console.print(f"[bold yellow]Rolling back to [bold cyan]transformers[/bold cyan]")
            self.model = None
            self.processor = None
            self._init_transformers_model(model_name)
            self.use_vllm = False

    def _init_transformers_model(self, model_name: str):
        """Initialize model using transformers library."""
        self.console.print(f"Implementing transformers model: [bold cyan]{model_name}[/bold cyan]")
        try:
            if torch.cuda.is_available():
                try:
                    from flash_attn import FlashMHA
                    attn_implementation = "flash_attention_2"
                    self.console.print("[bold green]Using flash attention for acceleration")
                except ImportError:
                    attn_implementation = "eager"
                    self.console.print("[bold yellow]Flash attention not available, using [bold cyan]eager[/bold cyan]")
            else:
                attn_implementation = "eager"
        except Exception as e:
            self.console.print(f"Error checking flash attention: [bold red]{str(e)}[/bold red]")
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


    def _init_system_prompt(self):
        """Initialize the system prompt."""
        self.console.print(f"[bold green]Agent ready ~ responding in [bold cyan]{self.language.upper()}[/bold cyan]")
        
        if self.model_config.supports_images:
            sys = [{"type": "text", "text": f"(Please response only in language {self.language.upper()})\n\n" + self.system_prompt}]
        else:
            sys = f"(Please response only in language {self.language.upper()})\n\n" + self.system_prompt
        self.context.extend([{
            "role": "system",
            "content": sys
        }])

    def clear_context(self):
        if hasattr(self, 'context'):
            self.context = [self.system_prompt] if hasattr(self, 'system_prompt') and self.system_prompt else []
        
        if hasattr(self, 'tasks'):
            for task in self.tasks.values():
                if task.get('image_path'):
                    try:
                        os.remove(task['image_path'])
                    except (FileNotFoundError, PermissionError, OSError) as e:
                        if hasattr(self, 'console'):
                            self.console.print(f"[bold red]Failed to delete temporary file [bold cyan]{task['image_path']}: {str(e)}[/bold cyan]")
            self.tasks = {}
        
        if hasattr(self, 'tmp_dir') and os.path.exists(self.tmp_dir):
            for file in os.listdir(self.tmp_dir):
                try:
                    os.remove(os.path.join(self.tmp_dir, file))
                except (FileNotFoundError, PermissionError, OSError) as e:
                    if hasattr(self, 'console'):
                        self.console.print(f"[bold red]Failed to delete temporary file [bold cyan]{file}: {str(e)}[/bold cyan]")


    def __del__(self):
        try:
            self.clear_context()
        except Exception as e:
            self.console.print(f"[bold red]Error during cleanup: [bold cyan]{str(e)}[/bold cyan]")

    @abstractmethod
    def generate_response(self, prompt: str, img_path: Optional[str] = None) -> dict:
        pass

    @abstractmethod
    def _process_task(self, job_id: str) -> dict:
        pass
