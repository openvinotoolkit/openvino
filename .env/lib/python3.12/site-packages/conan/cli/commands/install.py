import os

from conan.api.output import ConanOutput
from conan.cli import make_abs_path
from conan.cli.args import common_graph_args, validate_common_graph_args
from conan.cli.command import conan_command
from conan.cli.formatters.graph import format_graph_json
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_packages, print_graph_basic


@conan_command(group="Consumer", formatters={"json": format_graph_json})
def install(conan_api, parser, *args):
    """
    Install the requirements specified in a recipe (conanfile.py or conanfile.txt).

    It can also be used to install packages without a conanfile, using the
    --requires and --tool-requires arguments.

    If any requirement is not found in the local cache, it will iterate the remotes
    looking for it. When the full dependency graph is computed, and all dependencies
    recipes have been found, it will look for binary packages matching the current settings.
    If no binary package is found for some or several dependencies, it will error,
    unless the '--build' argument is used to build it from source.

    After installation of packages, the generators and deployers will be called.
    """
    common_graph_args(parser)
    parser.add_argument("-g", "--generator", action="append", help='Generators to use')
    parser.add_argument("-of", "--output-folder",
                        help='The root output folder for generated and build files')
    parser.add_argument("-d", "--deployer", action="append",
                        help="Deploy using the provided deployer to the output folder. "
                             "Built-in deployers: 'full_deploy', 'direct_deploy', 'runtime_deploy'")
    parser.add_argument("--deployer-folder",
                        help="Deployer output folder, base build folder by default if not set")
    parser.add_argument("--deployer-package", action="append",
                        help="Execute the deploy() method of the packages matching "
                             "the provided patterns")
    parser.add_argument("--build-require", action='store_true', default=False,
                        help='Whether the provided path is a build-require')
    parser.add_argument("--envs-generation", default=None, choices=["false"],
                        help="Generation strategy for virtual environment files for the root")
    args = parser.parse_args(*args)
    validate_common_graph_args(args)
    cwd = os.getcwd()

    deps_graph, lockfile, install_error = _run_install_command(conan_api, args, cwd)

    # Update lockfile if necessary
    lockfile = conan_api.lockfile.update_lockfile(lockfile, deps_graph, args.lockfile_packages,
                                                  clean=args.lockfile_clean)
    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, cwd)
    return {"graph": deps_graph,
            "conan_api": conan_api,
            "conan_error": install_error}


def _run_install_command(conan_api, args, cwd, return_install_error=True):
    """
    This method should not be imported as-is, it is internal to the installation process and
    its signature might change without warning.

    Users are however free to copy its code and adapt it to their needs, as an example of
    using the Conan API to perform an installation
    """
    # basic paths

    path = conan_api.local.get_conanfile_path(args.path, cwd, py=None) if args.path else None
    source_folder = os.path.dirname(path) if args.path else cwd
    output_folder = make_abs_path(args.output_folder, cwd) if args.output_folder else None

    # Basic collaborators: remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=path, cwd=cwd,
                                               partial=args.lockfile_partial, overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    # Graph computation (without installation of binaries)
    gapi = conan_api.graph
    if path:
        deps_graph = gapi.load_graph_consumer(path, args.name, args.version, args.user, args.channel,
                                              profile_host, profile_build, lockfile, remotes,
                                              args.update, is_build_require=args.build_require)
    else:
        deps_graph = gapi.load_graph_requires(args.requires, args.tool_requires, profile_host,
                                              profile_build, lockfile, remotes, args.update)
    print_graph_basic(deps_graph)
    deps_graph.report_graph_error()
    gapi.analyze_binaries(deps_graph, args.build, remotes, update=args.update, lockfile=lockfile)
    print_graph_packages(deps_graph)

    # Installation of binaries and consumer generators
    install_error = conan_api.install.install_binaries(deps_graph=deps_graph, remotes=remotes,
                                                       return_install_error=return_install_error)
    if not install_error:
        ConanOutput().title("Finalizing install (deploy, generators)")
        conan_api.install.install_consumer(deps_graph, args.generator, source_folder, output_folder,
                                           deploy=getattr(args, "deployer", None),
                                           deploy_package=getattr(args, "deployer_package", None),
                                           deploy_folder=getattr(args, "deployer_folder", None),
                                           envs_generation=getattr(args, "envs_generation", None))
        ConanOutput().success("Install finished successfully")

    return deps_graph, lockfile, install_error
