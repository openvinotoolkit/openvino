import os

from conan.api.output import ConanOutput
from conan.cli.command import conan_command, OnceArgument
from conan.cli.commands.create import test_package, _check_tested_reference_matches
from conan.cli.args import add_lockfile_args, add_common_install_arguments
from conan.cli.formatters.graph import format_graph_json
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_basic, print_graph_packages
from conan.api.model import RecipeReference


@conan_command(group="Creator", formatters={"json": format_graph_json})
def test(conan_api, parser, *args):
    """
    Test a package from a test_package folder.
    """
    parser.add_argument("path", action=OnceArgument,
                        help="Path to a test_package folder containing a conanfile.py. "
                             "Defaults to a 'test_package' folder in the current directory",
                        default="test_package", nargs='?')
    parser.add_argument("reference", action=OnceArgument,
                        help='Provide a package reference to test')
    add_common_install_arguments(parser)
    add_lockfile_args(parser)
    args = parser.parse_args(*args)

    cwd = os.getcwd()
    ref = RecipeReference.loads(args.reference)
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=True)
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial,
                                               overrides=overrides)
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)

    print_profiles(profile_host, profile_build)

    deps_graph = run_test(conan_api, path, ref, profile_host, profile_build, remotes, lockfile,
                          args.update, build_modes=args.build, tested_python_requires=ref)
    lockfile = conan_api.lockfile.update_lockfile(lockfile, deps_graph, args.lockfile_packages,
                                                  clean=args.lockfile_clean)
    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, cwd)

    return {"graph": deps_graph,
            "conan_api": conan_api}


def run_test(conan_api, path, ref, profile_host, profile_build, remotes, lockfile, update,
             build_modes, tested_python_requires=None, build_modes_test=None, tested_graph=None):
    root_node = conan_api.graph.load_root_test_conanfile(path, ref,
                                                         profile_host, profile_build,
                                                         remotes=remotes,
                                                         update=update,
                                                         lockfile=lockfile,
                                                         tested_python_requires=tested_python_requires)
    out = ConanOutput()
    out.title("Launching test_package")
    deps_graph = conan_api.graph.load_graph(root_node, profile_host=profile_host,
                                            profile_build=profile_build,
                                            lockfile=lockfile,
                                            remotes=remotes,
                                            update=update,
                                            check_update=update)
    print_graph_basic(deps_graph)
    deps_graph.report_graph_error()

    conan_api.graph.analyze_binaries(deps_graph, build_modes, remotes=remotes, update=update,
                                     lockfile=lockfile, build_modes_test=build_modes_test,
                                     tested_graph=tested_graph)
    print_graph_packages(deps_graph)

    conan_api.install.install_binaries(deps_graph=deps_graph, remotes=remotes)
    _check_tested_reference_matches(deps_graph, ref, out)
    test_package(conan_api, deps_graph, path)
    return deps_graph
