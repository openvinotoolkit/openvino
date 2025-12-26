import os
import tempfile

from conan.api.output import ConanOutput, LEVEL_STATUS, Color, LEVEL_ERROR, LEVEL_QUIET
from conan.cli.args import common_graph_args, validate_common_graph_args
from conan.cli.command import conan_command
from conan.cli.commands.install import _run_install_command
from conan.errors import ConanException


@conan_command(group="Consumer")
def run(conan_api, parser, *args):
    """
    (Experimental) Run a command given a set of requirements from a recipe or from command line.
    """
    common_graph_args(parser)
    parser.add_argument("command", help="Command to run")
    parser.add_argument("--context", help="Context to use, by default both contexts are activated "
                                          "if not specified",
                        choices=["host", "build"], default=None)
    parser.add_argument("--build-require", action='store_true', default=False,
                        help='Whether the provided path is a build-require')
    args = parser.parse_args(*args)
    validate_common_graph_args(args)
    cwd = os.getcwd()

    ConanOutput().info("Installing and building dependencies, this might take a while...",
                       fg=Color.BRIGHT_MAGENTA)
    previous_log_level = ConanOutput.get_output_level()
    if previous_log_level == LEVEL_STATUS:
        ConanOutput.set_output_level(LEVEL_QUIET)

    with tempfile.TemporaryDirectory("conanrun") as tmpdir:
        # Default values for install
        setattr(args, "output_folder", tmpdir)
        setattr(args, "generator", [])
        try:
            deps_graph, lockfile, _ = _run_install_command(conan_api, args, cwd,
                                                           return_install_error=False)
        except ConanException as e:
            ConanOutput.set_output_level(previous_log_level)
            ConanOutput().error("Error installing the dependencies. To debug this, you can either:\n"
                                " - Re-run the command with increased verbosity (-v, -vv)\n"
                                " - Run 'conan install' first to ensure dependencies are installed, "
                                "or to see errors during installation\n")
            raise e

        context_env_map = {
            "build": "conanbuild",
            "host": "conanrun",
        }
        envfiles = list(context_env_map.values()) if args.context is None \
            else [context_env_map.get(args.context)]
        ConanOutput.set_output_level(LEVEL_ERROR)
        deps_graph.root.conanfile.run(args.command, cwd=cwd, env=envfiles)
