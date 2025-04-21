import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.status import Status
from contextlib import contextmanager
from agent_engine.utils import load_config
from agent_engine.utils import import_class
from agent_engine import __version__

class BaseWorkflow(ABC):
    def __init__(self, config: str):
        self.console = Console()
        self.agent = None
        self._live_context = None
        self._load_config(config)
        self._init_agent_class()
        self._init_agents()

    def execute(self, *args, **kwargs) -> Any:
        workflow_type = self.cfg.raw.get("workflow", {}).get("type", "Unknown Workflow")
        self.console.print(Panel(f"[bold blue][WORKFLOW] Starting: {workflow_type}[/bold blue]", expand=False))
        
        try:
            self.console.print("[bold yellow][WORKFLOW] Pre-execution initialization...[/bold yellow]")
            self._pre_execute()
            result = self._execute(*args, **kwargs)
            return result
        except Exception as e:
            self.handle_error(e)
            raise
        finally:
            self.console.print("[bold yellow][WORKFLOW] Post-execution...[/bold yellow]")
            self._post_execute()
            self.console.print("[bold yellow][WORKFLOW] Cleaning up resources...[/bold yellow]")
            self.cleanup()
            self.console.print(Panel("[bold green][WORKFLOW] Completed[/bold green]", expand=False))

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        pass
    
    def _init_agents(self):
        pass
    
    def _load_config(self, config: str):
        """Load the configuration file."""
        self.cfg = load_config(config)
        
        config_table = Table(title="Configuration Details")
        config_table.add_column("Key", style="white", no_wrap=True)
        config_table.add_column("Value", style="grey50")
        
        for key, value in self.cfg.raw.items():
            if isinstance(value, dict):
                value = yaml.dump(value, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
            config_table.add_row(key, str(value))
        
        self.console.print(config_table)
        self.console.print(Panel("[bold green][WORKFLOW] Configuration Loaded[/bold green]", expand=False))
        text = f"""
╭──────  AGENT ENGINE  ─────╮╮
│  ░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒  ││
╰────────── {__version__} ──────────╯╯
"""
        self.console.print(text, style="bold green")
            
    def _init_agent_class(self):
        if self.cfg.workflow.agent.type == "multi_agents":
            self.agent_class = []
            for member_cfg in self.cfg.workflow.agent.members:
                self.agent_class.append(import_class(member_cfg["type"]))
        else:
            self.agent_class = import_class(self.cfg.workflow.agent.type)

    @contextmanager
    def _live_display(self, live_type="status", message=None):
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
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                self._live_context = progress
                yield progress
        else:
            raise ValueError("Unsupported live_type. Choose 'status' or 'progress'.")
        self._live_context = None

    def _pre_execute(self) -> None:
        pass

    def _post_execute(self) -> None:
        pass

    def handle_error(self, error: Exception) -> None:
        error_message = Syntax(str(error), "python", theme="monokai", line_numbers=False)
        self.console.print(Panel(error_message, title="[bold red][WORKFLOW] Error Occurred[/bold red]", expand=False))

    def cleanup(self) -> None:
        if hasattr(self, "_live_context") and self._live_context is not None:
            self._live_context.stop()
            self._live_context = None
        
        if self.agent:
            if isinstance(self.agent, list):
                for single_agent in self.agent:
                    if hasattr(single_agent, "chat_engine") and hasattr(single_agent.chat_engine, "clear_context"):
                        single_agent.chat_engine.clear_context()
            else:
                if hasattr(self.agent, "chat_engine") and hasattr(self.agent.chat_engine, "clear_context"):
                    self.agent.chat_engine.clear_context()

    def __del__(self):
        if self._live_context is not None:
            self.cleanup()
