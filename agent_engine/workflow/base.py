import yaml
from pathlib import Path
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
from agent_engine.utils.warpper import VerboseConsoleWrapper

class BaseWorkflow(ABC):
    """_summary_

    Args:
        config (str): Path to the workflow configuration file.
        
    Raises:
        FileNotFoundError: If the default configuration file is not found.
        ValueError: If the provided configuration file is invalid.
        
    This class serves as a base for creating different workflows in the agent engine.
    It handles loading the configuration, initializing agents, and executing the workflow.
    The workflow execution process includes pre-execution initialization, main execution,
    and post-execution cleanup. It also provides a live display for monitoring the workflow status.
    """
    def __init__(self, config: str):
        self.console = VerboseConsoleWrapper(Console(), role="WORKFLOW")
        self.agent = None
        self.default_config_path = str(Path(__file__).parent.parent.parent / "configs" / "base.yaml")
        self._live_context = None
        self._load_config(config)
        self._init_agent_class()
        self._init_agents()

    def execute(self, *args, **kwargs) -> Any:
        """_summary_

        Args:
            *args: Positional arguments for the workflow execution.
            **kwargs: Keyword arguments for the workflow execution.
            
        This method is responsible for executing the workflow. It handles pre-execution
        and post-execution tasks, as well as error handling. It also provides a live display
        for monitoring the workflow status.
        """
        workflow_type = self.cfg.raw.get("workflow", {}).get("type", "Unknown Workflow")
        self.console.print(Panel(f"Starting: [bold cyan]{workflow_type}[/bold cyan]", expand=False))
        
        try:
            self.console.print("[bold cyan]Pre-execution...", end="")
            self._pre_execute()
            self.console.print("[bold green] ✔", verbose=False)
            self.console.print("[bold cyan]Processing...", end="")
            result = self._execute(*args, **kwargs)
            self.console.print("[bold green] ✔", verbose=False)
            return result
        except Exception as e:
            self.handle_error(e)
            raise
        finally:
            self.console.print("[bold cyan]Post-execution...", end="")
            self._post_execute()
            self.console.print("[bold green] ✔", verbose=False)
            self.console.print("[bold cyan]Cleaning up resources...", end="")
            self.cleanup()
            self.console.print("[bold green] ✔", verbose=False)
            self.console.print(Panel("[bold green]Completed", expand=False))

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Any:
        pass
    
    def _init_agents(self):
        pass
    
    def _load_config(self, config: str):
        self.cfg = load_config(config)

        try:
            default_cfg = load_config(self.default_config_path)
        except FileNotFoundError:
            default_cfg = None
            self.console.print(f"[bold yellow]Default config not found at [bold cyan]{self.default_config_path}[/bold cyan], skipping...")

        if default_cfg:
            for key, value in default_cfg.raw.items():
                if key not in self.cfg.raw:
                    self.cfg.raw[key] = value

        config_table = Table(title="\nConfiguration Details")
        config_table.add_column("Key", style="white", no_wrap=True)
        config_table.add_column("Value", style="grey50")

        for key, value in self.cfg.raw.items():
            if isinstance(value, dict):
                value = yaml.dump(value, allow_unicode=True, sort_keys=False, default_flow_style=False).strip()
            config_table.add_row(key, str(value))

        self.console.print(config_table, verbose=False)
        self.console.print(Panel("[bold green]Configuration Loaded", expand=False))

        self.console.print(f"""
╭──────  AGENT ENGINE  ─────╮╮
│  ░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒▓░▒  ││  Help you creating things better ~
╰────────── [bold green]{__version__}[/bold green] ──────────╯╯
""", verbose=False)

            
    def _init_agent_class(self):
        if self.cfg.workflow.agent.type == "multi_agents":
            self.agent_class = []
            for member_cfg in self.cfg.workflow.agent.members:
                self.agent_class.append(import_class(member_cfg["type"]))
        else:
            self.agent_class = import_class(self.cfg.workflow.agent.type)

    @contextmanager
    def bar(self, live_type="status", message=None):
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
        self.console.print(Panel(error_message, title="[bold red]Error Occurred", expand=False))

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
