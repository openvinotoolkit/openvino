import os

from conan.api.output import ConanOutput
from conan.cli.command import conan_command
from conan.cli.formatters.graph import format_graph_json
from conan.cli import make_abs_path
from conan.cli.args import add_lockfile_args, add_common_install_arguments, add_reference_args
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_packages, print_graph_basic


@conan_command(group='Creator', formatters={"json": format_graph_json})
def build(conan_api, parser, *args):
    """
    Install dependencies and call the build() method.
    """
    parser.add_argument("path",
                        help='Path to a python-based recipe file or a folder '
                             'containing a conanfile.py recipe. conanfile.txt '
                             'cannot be used with conan build. '
                             'Defaults to current directory',
                        default=".", nargs='?')
    add_reference_args(parser)
    parser.add_argument("-g", "--generator", action="append", help='Generators to use')
    parser.add_argument("-of", "--output-folder",
                        help='The root output folder for generated and build files')
    parser.add_argument("-d", "--deployer", action="append",
                        help="Deploy using the provided deployer to the output folder. "
                             "Built-in deployers: 'full_deploy', 'direct_deploy', 'runtime_deploy'")
    parser.add_argument("--deployer-folder",
                        help="Deployer output folder, base build folder by default if not set")
    parser.add_argument("--build-require", action='store_true', default=False,
                        help='Whether the provided path is a build-require')
    parser.add_argument("--envs-generation", default=None, choices=["false"],
                        help="Generation strategy for virtual environment files for the root")
    add_common_install_arguments(parser)
    add_lockfile_args(parser)
    args = parser.parse_args(*args)

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=True)
    source_folder = os.path.dirname(path)
    output_folder = make_abs_path(args.output_folder, cwd) if args.output_folder else None
    deployer_folder = make_abs_path(args.deployer_folder, cwd) if args.deployer_folder else None

    # Basic collaborators: remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=path, cwd=cwd,
                                               partial=args.lockfile_partial, overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    deps_graph = conan_api.graph.load_graph_consumer(path, args.name, args.version,
                                                     args.user, args.channel,
                                                     profile_host, profile_build, lockfile, remotes,
                                                     args.update,
                                                     is_build_require=args.build_require)
    print_graph_basic(deps_graph)
    deps_graph.report_graph_error()
    conan_api.graph.analyze_binaries(deps_graph, args.build, remotes=remotes, update=args.update,
                                     lockfile=lockfile)
    print_graph_packages(deps_graph)

    out = ConanOutput()
    conan_api.install.install_binaries(deps_graph=deps_graph, remotes=remotes)

    out.title("Finalizing install (deploy, generators)")
    conan_api.install.install_consumer(deps_graph, args.generator, source_folder, output_folder,
                                       deploy=args.deployer, deploy_folder=deployer_folder,
                                       envs_generation=args.envs_generation)

    out.title("Calling build()")
    conanfile = deps_graph.root.conanfile
    conan_api.local.build(conanfile)

    lockfile = conan_api.lockfile.update_lockfile(lockfile, deps_graph, args.lockfile_packages,
                                                  clean=args.lockfile_clean)
    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, source_folder)
    return {"graph": deps_graph}
