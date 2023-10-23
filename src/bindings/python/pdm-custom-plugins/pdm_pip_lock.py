import platform

from pdm.cli.commands.lock import Command as BaseCommand
from pdm.core import Core


class PipLockCommand(BaseCommand):
    """
    This class is used to run the ``pdm lock`` command in a custom mode.

    The command will use the python version of the currently active environment.
    """

    description = BaseCommand.__doc__  # TODO: extend docs from Base
    name = "pip-lock"  # TODO: consider changing to "lock" to override original one

    def handle(self, project, options):
        # Pass project to the function to reuse options available for Project class.
        # For style refer to: https://github.com/pdm-project/pdm/blob/main/src/pdm/termui.py
        def note(project, style: str, message: str) -> None:
            if not project.is_global:
                project.core.ui.echo(message, style=style, err=True)

        # options.cross_platform = True  # set --no-cross-platform if False

        # Get current environment version and apply to current install session.
        project.pyproject.metadata["requires-python"] = f"=={platform.python_version()}"
        note(project, "warning", f"Python version has been overriden by the user.")

        note(
            project,
            "info",
            f"Custom locking with Python version set to: {project.pyproject.metadata.get('requires-python', '')}",
        )

        # Run `pdm lock` itself.
        super().handle(project, options)


def register(core: Core) -> None:
    core.register_command(PipLockCommand)
