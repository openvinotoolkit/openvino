import json
import os

from conan.api.conan_api import ConanAPI
from conan.api.model import RecipeReference
from conan.api.output import ConanOutput, cli_out_write
from conan.api.subapi.workspace import WorkspaceAPI
from conan.cli import make_abs_path
from conan.cli.args import add_reference_args, add_common_install_arguments, add_lockfile_args
from conan.cli.command import conan_command, conan_subcommand
from conan.cli.commands.list import print_serial
from conan.cli.formatters.graph import format_graph_json
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_packages, print_graph_basic
from conan.errors import ConanException
from conan.internal.graph.install_graph import ProfileArgs


@conan_subcommand(formatters={"text": cli_out_write})
def workspace_root(conan_api: ConanAPI, parser, subparser, *args):  # noqa
    """
    Return the folder containing the conanws.py/conanws.yml workspace file
    """
    parser.parse_args(*args)
    ws = conan_api.workspace
    return ws.folder()


@conan_subcommand()
def workspace_open(conan_api: ConanAPI, parser, subparser, *args):
    """
    Open specific references
    """
    subparser.add_argument("reference",
                           help="Open this package source repository")
    group = subparser.add_mutually_exclusive_group()
    group.add_argument("-r", "--remote", action="append", default=None,
                       help='Look in the specified remote or remotes server')
    group.add_argument("-nr", "--no-remote", action="store_true",
                       help='Do not use remote, resolve exclusively in the cache')
    args = parser.parse_args(*args)
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    cwd = os.getcwd()
    conan_api.workspace.open(args.reference, remotes=remotes, cwd=cwd)


@conan_subcommand()
def workspace_add(conan_api: ConanAPI, parser, subparser, *args):
    """
    Add packages to current workspace
    """
    subparser.add_argument('path',  nargs="?",
                           help='Path to the package folder in the user workspace')
    add_reference_args(subparser)
    subparser.add_argument("--ref", help="Open and add this reference")
    subparser.add_argument("-of", "--output-folder",
                           help='The root output folder for generated and build files')
    group = subparser.add_mutually_exclusive_group()
    group.add_argument("-r", "--remote", action="append", default=None,
                       help='Look in the specified remote or remotes server')
    group.add_argument("-nr", "--no-remote", action="store_true",
                       help='Do not use remote, resolve exclusively in the cache')
    args = parser.parse_args(*args)
    if args.path and args.ref:
        raise ConanException("Do not use both 'path' and '--ref' argument")
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    cwd = os.getcwd()
    path = args.path
    if args.ref:
        # TODO: Use path here to open in this path
        path = conan_api.workspace.open(args.ref, remotes, cwd=cwd)
    ref = conan_api.workspace.add(path,
                                  args.name, args.version, args.user, args.channel,
                                  cwd, args.output_folder, remotes=remotes)
    ConanOutput().success("Reference '{}' added to workspace".format(ref))


@conan_subcommand()
def workspace_complete(conan_api: ConanAPI, parser, subparser, *args):
    """
    Complete the workspace, opening or adding intermediate packages to it that have
    requirements to other packages in the workspace.
    """
    add_common_install_arguments(subparser)
    add_lockfile_args(subparser)
    args = parser.parse_args(*args)
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    # The lockfile by default if not defined will be read from the root workspace folder
    ws_folder = conan_api.workspace.folder()
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=ws_folder,
                                               cwd=None, partial=args.lockfile_partial,
                                               overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    ConanOutput().box("Workspace completing intermediate packages")
    conan_api.workspace.complete(profile_host, profile_build, lockfile, remotes, update=args.update)


@conan_subcommand()
def workspace_remove(conan_api: ConanAPI, parser, subparser, *args):
    """
    Remove packages from the current workspace
    """
    subparser.add_argument('path',
                           help='Path to the package folder in the user workspace')
    args = parser.parse_args(*args)
    removed = conan_api.workspace.remove(make_abs_path(args.path))
    ConanOutput().info(f"Removed from workspace: {removed}")


def _print_json(data):
    results = data["info"]
    myjson = json.dumps(results, indent=4)
    cli_out_write(myjson)


def _print_workspace_info(data):
    ret = []
    for package_info in data["info"]["packages"]:
        ret.append(f"- path: {package_info['path']}")
        ref = package_info.get('ref')
        if ref:
            ret.append(f"  ref: {ref}")
    data["info"]["packages"] = ret or "(empty)"
    print_serial(data["info"])


@conan_subcommand(formatters={"text": _print_workspace_info, "json": _print_json})
def workspace_info(conan_api: ConanAPI, parser, subparser, *args):  # noqa
    """
    Display info for current workspace
    """
    parser.parse_args(*args)
    return {"info": conan_api.workspace.info()}


@conan_subcommand()
def workspace_build(conan_api: ConanAPI, parser, subparser, *args):
    """
    Call "conan build" for packages in the workspace, in the right order
    """
    _install_build(conan_api, parser, subparser, True, *args)


@conan_subcommand()
def workspace_install(conan_api: ConanAPI, parser, subparser, *args):
    """
    Call "conan install" for packages in the workspace, in the right order
    """
    _install_build(conan_api, parser, subparser, False, *args)


def _install_build(conan_api: ConanAPI, parser, subparser, build, *args):
    subparser.add_argument("--pkg", action="append", help='Define specific packages')
    add_common_install_arguments(subparser)
    add_lockfile_args(subparser)
    args = parser.parse_args(*args)
    # Basic collaborators: remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    # The lockfile by default if not defined will be read from the root workspace folder
    ws_folder = conan_api.workspace.folder()
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=ws_folder,
                                               cwd=None, partial=args.lockfile_partial,
                                               overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    buildmode = args.build
    if build and (not buildmode or "editable" not in buildmode):
        ConanOutput().info("Adding '--build=editable' as build mode")
        buildmode = buildmode or []
        buildmode.append("editable")

    all_editables = conan_api.workspace.packages()
    packages = conan_api.workspace.select_packages(args.pkg)
    ConanOutput().box("Workspace computing the build order")
    install_order = conan_api.workspace.build_order(packages, profile_host, profile_build, buildmode,
                                                    lockfile, remotes, args, update=args.update)

    msg = "build" if build else "install"
    ConanOutput().box(f"Workspace {msg}ing each package")
    order = install_order.install_build_order()

    profile_args = ProfileArgs.from_args(args)
    for level in order["order"]:
        for elem in level:
            ref = RecipeReference.loads(elem["ref"])
            for package_level in elem["packages"]:
                for package in package_level:
                    ws_pkg = all_editables.get(ref)
                    is_editable = package["binary"] in ("Editable", "EditableBuild")
                    if ws_pkg is None:
                        if is_editable or package["binary"] == "Build": # Build external to Workspace
                            cmd = f'install {package["build_args"]} {profile_args}'
                            ConanOutput().box(f"Workspace building external {ref}")
                            ConanOutput().info(f"Command: {cmd}\n")
                            conan_api.command.run(cmd)
                    else:
                        path = ws_pkg["path"]
                        output_folder = ws_pkg.get("output_folder")
                        build_arg = "--build-require" if package["context"] == "build" else ""
                        ref_args = " ".join(f"--{k}={getattr(ref, k)}"
                                            for k in ("name", "version", "user", "channel")
                                            if getattr(ref, k, None))
                        of_arg = f'-of="{output_folder}"' if output_folder else ""
                        # TODO: Missing --lockfile-overrides arg here
                        command = "build" if build else "install"
                        cmd = f'{command} "{path}" {profile_args} {build_arg} {ref_args} {of_arg}'
                        ConanOutput().box(f"Workspace {command}: {ref}")
                        ConanOutput().info(f"Command: {cmd}\n")
                        conan_api.command.run(cmd)


@conan_subcommand(formatters={"json": format_graph_json})
def workspace_super_install(conan_api: ConanAPI, parser, subparser, *args):
    """
    Install the workspace as a monolith, installing only external dependencies to the workspace,
    generating a single result (generators, etc) for the whole workspace.
    """
    subparser.add_argument("--pkg", action="append", help='Define specific packages')
    subparser.add_argument("-g", "--generator", action="append", help='Generators to use')
    subparser.add_argument("-of", "--output-folder",
                           help='The root output folder for generated and build files')
    subparser.add_argument("-d", "--deployer", action="append",
                           help="Deploy using the provided deployer to the output folder. "
                                "Built-in deployers: 'full_deploy', 'direct_deploy', "
                                "'runtime_deploy'")
    subparser.add_argument("--deployer-folder",
                           help="Deployer output folder, base build folder by default if not set")
    subparser.add_argument("--deployer-package", action="append",
                           help="Execute the deploy() method of the packages matching "
                                "the provided patterns")
    subparser.add_argument("--envs-generation", default=None, choices=["false"],
                           help="Generation strategy for virtual environment files for the root")
    add_common_install_arguments(subparser)
    add_lockfile_args(subparser)
    args = parser.parse_args(*args)
    # Basic collaborators: remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    # The lockfile by default if not defined will be read from the root workspace folder
    ws_folder = conan_api.workspace.folder()
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=ws_folder,
                                               cwd=None,
                                               partial=args.lockfile_partial, overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    # Build a dependency graph with all editables as requirements
    requires = conan_api.workspace.select_packages(args.pkg)
    deps_graph = conan_api.graph.load_graph_requires(requires, [],
                                                     profile_host, profile_build, lockfile,
                                                     remotes, update=args.update)
    deps_graph.report_graph_error()
    print_graph_basic(deps_graph)

    # Collapsing the graph
    ws_graph = conan_api.workspace.super_build_graph(deps_graph, profile_host, profile_build)
    ConanOutput().subtitle("Collapsed graph")
    print_graph_basic(ws_graph)
    conan_api.graph.analyze_binaries(ws_graph, args.build, remotes=remotes, update=args.update,
                                     lockfile=lockfile)
    print_graph_packages(ws_graph)
    conan_api.install.install_binaries(deps_graph=ws_graph, remotes=remotes)
    ConanOutput().title("Finalizing install (deploy, generators)")
    output_folder = make_abs_path(args.output_folder) if args.output_folder else None
    conan_api.install.install_consumer(ws_graph, args.generator, ws_folder, output_folder,
                                       deploy=args.deployer, deploy_package=args.deployer_package,
                                       deploy_folder=args.deployer_folder,
                                       envs_generation=args.envs_generation)
    ConanOutput().success("Install finished successfully")

    return {"graph": ws_graph,
            "conan_api": conan_api}


@conan_subcommand()
def workspace_clean(conan_api: ConanAPI, parser, subparser, *args):  # noqa
    """
    Clean the temporary build folders when possible
    """
    parser.parse_args(*args)
    conan_api.workspace.clean()


@conan_subcommand()
def workspace_init(conan_api: ConanAPI, parser, subparser, *args):
    """
    Clean the temporary build folders when possible
    """
    subparser.add_argument("path", nargs="?", default=os.getcwd(),
                           help="Path to a folder where the workspace will be initialized. "
                                "Defaults to the current directory")
    args = parser.parse_args(*args)
    conan_api.workspace.init(args.path)


@conan_subcommand()
def workspace_create(conan_api: ConanAPI, parser, subparser, *args):
    """
    Call "conan create" for packages in the workspace, in the correct order.
    Packages will be created in the Conan cache, not locally
    """
    subparser.add_argument("--pkg", action="append", help='Define specific packages')
    add_common_install_arguments(subparser)
    add_lockfile_args(subparser)
    args = parser.parse_args(*args)
    # Basic collaborators: remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    # The lockfile by default if not defined will be read from the root workspace folder
    ws_folder = conan_api.workspace.folder()
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile, conanfile_path=ws_folder,
                                               cwd=None,
                                               partial=args.lockfile_partial, overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    build_mode = args.build if args.build else []
    ConanOutput().box("Exporting workspace recipes to Conan cache")
    exported_refs = conan_api.workspace.export()
    build_mode.extend(f"missing:{r}" for r in exported_refs)

    all_packages = conan_api.workspace.packages()
    packages = conan_api.workspace.select_packages(args.pkg)

    # If we don't disable the workspace, then, the packages are not created in the Conan cache,
    # but locally in the user folders, are they are intercepted as editables
    conan_api.workspace.enable(False)

    install_order = conan_api.workspace.build_order(packages, profile_host, profile_build,
                                                    build_mode, lockfile, remotes, args,
                                                    update=args.update)

    ConanOutput().box("Workspace creating each package")
    order = install_order.install_build_order()

    profile_args = ProfileArgs.from_args(args)
    for level in order["order"]:
        for elem in level:
            ref = RecipeReference.loads(elem["ref"])
            for package_level in elem["packages"]:
                for package in package_level:
                    if package["binary"] not in ("Build", "EditableBuild"):
                        ConanOutput().info(f"Skip build for {ref} binary: {package['binary']}")
                        continue

                    if ref not in all_packages:
                        # Build external to Workspace
                        cmd = f'install {package["build_args"]} {profile_args}'
                        ConanOutput().box(f"Workspace building external {ref}")
                        ConanOutput().info(f"Build command: {cmd}\n")
                        conan_api.command.run(cmd)
                    else:  # Package in workspace
                        path = packages[ref]["path"]
                        # TODO: Missing --lockfile-overrides arg here
                        build = "--build-require" if package["context"] == "build" else ""
                        ref_args = " ".join(f"--{k}={getattr(ref, k)}"
                                            for k in ("name", "version", "user", "channel")
                                            if getattr(ref, k, None))
                        cmd = f'create "{path}" {profile_args} {build} {ref_args}'
                        ConanOutput().box(f"Workspace create {ref}")
                        ConanOutput().info(f"Conan create command: {cmd}\n")
                        conan_api.command.run(cmd)


@conan_subcommand()
def workspace_source(conan_api: ConanAPI, parser, subparser, *args):
    """
    Call the source() method of packages in the workspace
    """
    subparser.add_argument("--pkg", action="append", help='Define specific packages')
    args = parser.parse_args(*args)

    remotes = conan_api.remotes.list()  # In case "python_requires" are needed
    packages = conan_api.workspace.select_packages(args.pkg)

    ConanOutput().box("Workspace getting sources")
    for pkg, info in packages.items():
        conan_api.local.source(info["path"], name=pkg.name, version=str(pkg.version),
                               user=pkg.user, channel=pkg.channel, remotes=remotes)


@conan_command(group="Consumer")
def workspace(conan_api, parser, *args):  # noqa
    """
    Manage Conan workspaces (group of packages in editable mode)
    """
    if (WorkspaceAPI.TEST_ENABLED or os.getenv("CONAN_WORKSPACE_ENABLE")) != "will_break_next":
        raise ConanException("Workspace command disabled without CONAN_WORKSPACE_ENABLE env var,"
                             "please read the docs about this 'incubating' feature")
