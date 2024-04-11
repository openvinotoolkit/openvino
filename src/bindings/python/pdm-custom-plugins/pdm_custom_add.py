import platform

from pdm.project.config import ConfigItem

# from pdm.cli.commands.base import BaseCommand
from pdm.cli.commands.add import Command as BaseCommand
from pdm.core import Core


CONFIG = {
    "python.version": ConfigItem(
        "Use this to override 'python-requires' of the project",
        default="",
    ),
}


class CustomAddCommand(BaseCommand):
    """
    This class is used to run the ``pdm add`` command in a custom mode.

    :param --python: This parameter can have three possible values:

        1. If set to 'env', the command will use the python version of the currently active environment.
        2. If set to a specific version (e.g. '>=3.10'), the command will use that version.
        3. If not provided, the command will read from the `python.version` config.

    If the `python.version` config is unset, the command will use the project's `requires-python` value.
    This command always prioritizes `--python` flag over the `python.version`.

    """

    description = BaseCommand.__doc__
    name = "custom-add"
    def add_arguments(self, parser):
        parser.add_argument(
            "--python", help="Python version to replace project's `requires-python`."
        )

        # Append `pdm install` attributes itself.
        super().add_arguments(parser)

    def handle(self, project, options):
        # Pass project to the function to reuse options available for Project class.
        # For style refer to: https://github.com/pdm-project/pdm/blob/main/src/pdm/termui.py
        def note(project, style: str, message: str) -> None:
            if not project.is_global:
                project.core.ui.echo(message, style=style, err=True)

        # Get version from either options or config. Prioritize `--python` flag.
        python_version = options.python or project.config.get("python.version", "")
        # Check if existing and apply to current install session.
        if python_version:
            if python_version == "env":
                project.pyproject.metadata[
                    "requires-python"
                ] = f"=={platform.python_version()}"
            else:
                project.pyproject.metadata["requires-python"] = python_version
            note(project, "warning", f"Python version has been overriden by the user.")

        note(
            project,
            "info",
            f"Custom add with Python version set to: {project.pyproject.metadata.get('requires-python', '')}",
        )

        # Run `pdm add` itself.
        super().handle(project, options)


def register(core: Core) -> None:
    core.register_command(CustomAddCommand)
    for k, v in CONFIG.items():
        core.add_config(k, v)
