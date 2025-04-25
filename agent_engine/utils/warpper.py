from rich.panel import Panel
from rich.text import Text

class VerboseConsoleWrapper:
    def __init__(self, console, role="", role_color="bright_black"):
        self.console = console
        self.role = role
        self.role_color = role_color

    def _format_content(self, content):
        if isinstance(content, Panel):
            if isinstance(content.renderable, str):
                panel_content = Text.from_markup(content.renderable)
            else:
                panel_content = content.renderable
            new_content = Text.from_markup(
                f"[bold {self.role_color}][{self.role}][/bold {self.role_color}] "
            ) + panel_content
            return Panel(
                new_content,
                title=content.title,
                subtitle=content.subtitle,
                border_style=content.border_style,
                expand=content.expand,
                padding=content.padding,
                style=content.style,
            )
        else:
            return f"[bold {self.role_color}][{self.role}][/bold {self.role_color}] " + str(content)

    def print(self, *args, **kwargs):
        verbose = kwargs.pop("verbose", True)
        if args and verbose:
            args = (self._format_content(args[0]),) + args[1:]
        self.console.print(*args, **kwargs)

    def log(self, *args, **kwargs):
        verbose = kwargs.pop("verbose", True)
        if args and verbose:
            args = (self._format_content(args[0]),) + args[1:]
        self.console.log(*args, **kwargs)
