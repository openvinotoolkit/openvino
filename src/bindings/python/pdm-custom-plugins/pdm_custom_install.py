import platform

from pdm.project.config import ConfigItem

# from pdm.cli.commands.base import BaseCommand
from pdm.cli.commands.install import Command as BaseCommand
from pdm.core import Core


CONFIG = {
    "python.version": ConfigItem(
        "Use this to override 'python-requires' of the project",
        default="",
    ),
}


class CustomInstallCommand(BaseCommand):
    """
    This class is used to run the ``pdm install`` command in a custom mode.

    :param --python: This parameter can have three possible values:

        1. If set to 'env', the command will use the python version of the currently active environment.
        2. If set to a specific version (e.g. '>=3.10'), the command will use that version.
        3. If not provided, the command will read from the `python.version` config.

    If the `python.version` config is unset, the command will use the project's `requires-python` value.
    This command always prioritizes `--python` flag over the `python.version`.

    .. code::

        # To use active environment version:
        $ pdm custom-install --python env
        # To use range from 3.10.0 to 3.11.*:
        $ pdm custom-install --python ">3.9, <=3.11"
        # Works with interface of `pdm install`:
        $ pdm custom-install --python env --no-lock --no-self --dry-run

    :param --requirements: This parameter applies `--no-lock --no-self`.

    """

    description = BaseCommand.__doc__
    name = "custom-install"  # TODO: consider changing to "install" to override original one

    def add_arguments(self, parser):
        parser.add_argument(
            "--python", help="Python version to replace project's `requires-python`."
        )
        parser.add_argument(
            "--requirements",
            action="store_true",
            dest="only_requirements",
            help="Install requirements only. Applies `--no-lock` and `--no-self` automatically",
        )

        # Append `pdm install` attributes itself.
        super().add_arguments(parser)

    def handle(self, project, options):
        # Pass project to the function to reuse options available for Project class.
        # For style refer to: https://github.com/pdm-project/pdm/blob/main/src/pdm/termui.py
        def note(project, style: str, message: str) -> None:
            if not project.is_global:
                project.core.ui.echo(message, style=style, err=True)

        if options.only_requirements:
            note(project, "warning", f"Installing only requirements of the project.")
            project.enable_write_lockfile = False
            options.no_self = True

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
            f"Custom installation with Python version set to: {project.pyproject.metadata.get('requires-python', '')}",
        )

        # Probably not needed?
        # if options.groups:
        #     if ":all" in options.groups:
        #         options.groups += list(project.iter_groups())

        # Run `pdm install` itself.
        super().handle(project, options)


def register(core: Core) -> None:
    core.register_command(CustomInstallCommand)
    for k, v in CONFIG.items():
        core.add_config(k, v)
