import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.status import Status
from contextlib import contextmanager
from agent_engine.utils import agent_engine_version

class BaseWorkflow(ABC):
    """
    工作流抽象基类，用于协调多个代理（Agents）完成复杂任务。
    子类需实现 `_execute` 方法定义具体执行逻辑。
    """

    def __init__(self, config: str):
        self.console = Console()
        self._load_config(config)
        self._live_context = None  # 用于管理动态显示的上下文
        
    def _load_config(self, config: str):
        """Load the configuration file."""
        with open(config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        config_table = Table(title="Configuration Details")
        config_table.add_column("Key", style="white", no_wrap=True)
        config_table.add_column("Value", style="grey50")
        
        for key, value in self.cfg.items():
            if isinstance(value, dict):
                value = yaml.dump(value, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
            config_table.add_row(key, str(value))
        
        self.console.print(config_table)
        self.console.print(Panel("[bold green][WORKFLOW] Configuration Loaded[/bold green]", expand=False))
        text = f"""
╭──────  AGENT ENGINE  ─────╮╮
│  ░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒  ││
╰────────── {agent_engine_version} ─────────╯╯
"""
        self.console.print(text, style="bold green")
            
    def _init_agent(self):
        """初始化助手"""
        pass

    @contextmanager
    def _live_display(self, live_type="status", message=None):
        """统一管理动态显示的上下文"""
        if self._live_context is not None:
            raise RuntimeError("A live display is already active. Nested live displays are not allowed.")
        
        if live_type == "status":
            with Status(f"[bold yellow]{message or 'Executing Workflow...'}", spinner="dots") as status:
                self._live_context = status
                yield status
        elif live_type == "progress":
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
            ) as progress:
                self._live_context = progress
                yield progress
        else:
            raise ValueError("Unsupported live_type. Choose 'status' or 'progress'.")
        self._live_context = None

    def execute(self, *args, **kwargs) -> Any:
        """执行工作流，包含预处理、执行、后处理及错误处理"""
        workflow_type = self.cfg.get("workflow", {}).get("type", "Unknown Workflow")
        self.console.print(Panel(f"[bold blue][WORKFLOW] Starting: {workflow_type}[/bold blue]", expand=False))
        
        self.pre_execute()
        try:
            result = self._execute(*args, **kwargs)
            return result
        except Exception as e:
            self.handle_error(e)
            raise
        finally:
            self.cleanup()
            self.console.print(Panel("[bold green][WORKFLOW] Completed[/bold green]", expand=False))

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        """子类需实现的具体工作流逻辑"""
        pass

    def pre_execute(self) -> None:
        """执行前的初始化操作（可覆盖）"""
        self.console.print("[bold yellow][WORKFLOW] Pre-execution initialization...[/bold yellow]")

    def post_execute(self) -> None:
        """执行后的收尾操作（可覆盖）"""
        self.console.print("[bold yellow][WORKFLOW] Post-execution cleanup...[/bold yellow]")

    def handle_error(self, error: Exception) -> None:
        """全局错误处理（可覆盖）"""
        error_message = Syntax(str(error), "python", theme="monokai", line_numbers=False)
        self.console.print(Panel(error_message, title="[bold red][WORKFLOW] Error Occurred[/bold red]", expand=False))

    def cleanup(self) -> None:
        """资源清理，自动调用代理的清理方法"""
        self.console.print("[bold yellow][WORKFLOW] Cleaning up resources...[/bold yellow]")

    def __del__(self):
        # 避免在解释器关闭时调用 cleanup 导致异常
        if self._live_context is not None:
            self.cleanup()