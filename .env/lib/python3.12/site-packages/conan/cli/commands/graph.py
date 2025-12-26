import json
import os

from conan.api.output import ConanOutput, cli_out_write, Color
from conan.cli import make_abs_path
from conan.cli.args import common_graph_args, validate_common_graph_args
from conan.cli.command import conan_command, conan_subcommand
from conan.cli.commands.list import prepare_pkglist_compact, print_serial
from conan.cli.formatters.graph import format_graph_html, format_graph_json, format_graph_dot
from conan.cli.formatters.graph.build_order_html import format_build_order_html
from conan.cli.formatters.graph.graph_info_text import format_graph_info
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_packages, print_graph_basic
from conan.errors import ConanException


def explain_formatter_text(data):
    if "closest_binaries" in data:
        # To be able to reuse the print_list_compact method,
        # we need to wrap this in a MultiPackagesList
        pkglist = data["closest_binaries"]
        prepare_pkglist_compact(pkglist)
        # Now we make sure that if there are no binaries we will print something that makes sense
        for ref, ref_info in pkglist.items():
            for rrev, rrev_info in ref_info.items():
                if not rrev_info:
                    rrev_info["ERROR"] = "No package binaries exist"
        print_serial(pkglist)


def explain_formatter_json(data):
    myjson = json.dumps(data, indent=4)
    cli_out_write(myjson)


@conan_command(group="Consumer")
def graph(conan_api, parser, *args):  # noqa
    """
    Compute a dependency graph, without installing or building the binaries.
    """


def cli_build_order(result):
    # TODO: Very simple cli output, probably needs to be improved
    build_order = result["build_order"]
    build_order = build_order["order"] if isinstance(build_order, dict) else build_order
    for level in build_order:
        for item in level:
            # If this is a configuration order, it has no packages entry, each item is a package
            if 'packages' in item:
                for package_level in item['packages']:
                    for package in package_level:
                        cli_out_write(f"{item['ref']}:{package['package_id']} - {package['binary']}")
            else:
                cli_out_write(f"{item['ref']}:{item['package_id']} - {item['binary']}")


def json_build_order(result):
    cli_out_write(json.dumps(result["build_order"], indent=4))


@conan_subcommand(formatters={"text": cli_build_order, "json": json_build_order,
                              "html": format_build_order_html})
def graph_build_order(conan_api, parser, subparser, *args):
    """
    Compute the build order of a dependency graph.
    """
    common_graph_args(subparser)
    subparser.add_argument("--order-by", choices=['recipe', 'configuration'],
                           help='Select how to order the output, "recipe" by default if not set.')
    subparser.add_argument("--reduce", action='store_true', default=False,
                           help='Reduce the build order, output only those to build. Use this '
                                'only if the result will not be merged later with other build-order')
    args = parser.parse_args(*args)
    validate_common_graph_args(args)
    if args.order_by is None:
        ConanOutput().warning("Please specify --order-by argument", warn_tag="deprecated")

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=None) if args.path else None

    # Basic collaborators, remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial,
                                               overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)

    if path:
        deps_graph = conan_api.graph.load_graph_consumer(path, args.name, args.version,
                                                         args.user, args.channel,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.build, args.update)
    else:
        deps_graph = conan_api.graph.load_graph_requires(args.requires, args.tool_requires,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.build, args.update)
    print_graph_basic(deps_graph)
    deps_graph.report_graph_error()
    conan_api.graph.analyze_binaries(deps_graph, args.build, remotes=remotes, update=args.update,
                                     lockfile=lockfile)
    print_graph_packages(deps_graph)

    out = ConanOutput()
    out.title("Computing the build order")
    install_graph = conan_api.graph.build_order(deps_graph, args.order_by, args.reduce,
                                                profile_args=args)
    install_order_serialized = install_graph.install_build_order()
    if args.order_by is None:  # legacy
        install_order_serialized = install_order_serialized["order"]

    lockfile = conan_api.lockfile.update_lockfile(lockfile, deps_graph, args.lockfile_packages,
                                                  clean=args.lockfile_clean)
    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, cwd)

    return {"build_order": install_order_serialized,
            "conan_error": install_graph.get_errors()}


@conan_subcommand(formatters={"text": cli_build_order, "json": json_build_order,
                              "html": format_build_order_html})
def graph_build_order_merge(conan_api, parser, subparser, *args):  # noqa
    """
    Merge more than 1 build-order file.
    """
    subparser.add_argument("--file", nargs="?", action="append", help="Files to be merged")
    subparser.add_argument("--reduce", action='store_true', default=False,
                           help='Reduce the build order, output only those to build. Use this '
                                'only if the result will not be merged later with other build-order')
    args = parser.parse_args(*args)
    if not args.file or len(args.file) < 2:
        raise ConanException("At least 2 files are needed to be merged")

    files = [make_abs_path(f) for f in args.file]
    result = conan_api.graph.build_order_merge(files, args.reduce)

    install_order_serialized = result.install_build_order()
    if getattr(result, "legacy"):
        install_order_serialized = install_order_serialized["order"]
    return {"build_order": install_order_serialized,
            "conan_error": result.get_errors()}


@conan_subcommand(formatters={"text": format_graph_info,
                              "html": format_graph_html,
                              "json": format_graph_json,
                              "dot": format_graph_dot})
def graph_info(conan_api, parser, subparser, *args):
    """
    Compute the dependency graph and show information about it.
    """
    common_graph_args(subparser)
    subparser.add_argument("--check-updates", default=False, action="store_true",
                           help="Check if there are recipe updates")
    subparser.add_argument("--filter", action="append",
                           help="Show only the specified fields")
    subparser.add_argument("--package-filter", action="append",
                           help='Print information only for packages that match the patterns')
    subparser.add_argument("-d", "--deployer", action="append",
                           help="Deploy using the provided deployer to the output folder. "
                                "Built-in deployers: 'full_deploy', 'direct_deploy'. Deployers "
                                "will only deploy recipes, as 'conan graph info' do not retrieve "
                                "binaries")
    subparser.add_argument("-df", "--deployer-folder",
                           help="Deployer output folder, base build folder by default if not set")
    subparser.add_argument("--build-require", action='store_true', default=False,
                           help='Whether the provided reference is a build-require')
    args = parser.parse_args(*args)
    # parameter validation
    validate_common_graph_args(args)
    if args.format in ("html", "dot") and args.filter:
        raise ConanException(f"Formatted output '{args.format}' cannot filter fields")

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=None) if args.path else None

    # Basic collaborators, remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial,
                                               overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)
    print_profiles(profile_host, profile_build)

    if path:
        deps_graph = conan_api.graph.load_graph_consumer(path, args.name, args.version,
                                                         args.user, args.channel,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.update,
                                                         check_updates=args.check_updates,
                                                         is_build_require=args.build_require)
    else:
        deps_graph = conan_api.graph.load_graph_requires(args.requires, args.tool_requires,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.update,
                                                         check_updates=args.check_updates)
    print_graph_basic(deps_graph)
    if not deps_graph.error:
        conan_api.graph.analyze_binaries(deps_graph, args.build, remotes=remotes, update=args.update,
                                         lockfile=lockfile)
        print_graph_packages(deps_graph)

        conan_api.install.install_system_requires(deps_graph, only_info=True)
        conan_api.install.install_sources(deps_graph, remotes=remotes)

        lockfile = conan_api.lockfile.update_lockfile(lockfile, deps_graph, args.lockfile_packages,
                                                      clean=args.lockfile_clean)
        conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, cwd)
        if args.deployer:
            base_folder = args.deployer_folder or os.getcwd()
            conan_api.install.deploy(deps_graph, args.deployer, None, base_folder)

    warn_msg = None
    missing = set(str(n.ref.name) for n in deps_graph.nodes if n.binary == "Missing")
    invalid = set(str(n.ref.name) for n in deps_graph.nodes if n.binary == "Invalid")
    if missing or invalid:
        warn_msg = "There are some error(s) in the graph:"
        if missing:
            warn_msg += f"\n    - Missing packages: {', '.join(missing)}"
        if invalid:
            warn_msg += f"\n    - Invalid packages: {', '.join(invalid)}"

    return {"graph": deps_graph,
            "field_filter": args.filter,
            "package_filter": args.package_filter,
            "conan_api": conan_api,
            "conan_error": str(deps_graph.error) if deps_graph.error else None,
            "conan_warning": warn_msg}


@conan_subcommand(formatters={"text": explain_formatter_text,
                              "json": explain_formatter_json})
def graph_explain(conan_api, parser,  subparser, *args):
    """
    Explain what is wrong with the dependency graph, like report missing binaries closest
    alternatives, trying to explain why the existing binaries do not match
    """
    common_graph_args(subparser)
    subparser.add_argument("--check-updates", default=False, action="store_true",
                           help="Check if there are recipe updates")
    subparser.add_argument("--build-require", action='store_true', default=False,
                           help='Whether the provided reference is a build-require')
    subparser.add_argument('--missing', nargs="?",
                           help="A pattern in the form 'pkg/version#revision:package_id#revision', "
                                "e.g: \"zlib/1.2.13:*\" means all binaries for zlib/1.2.13. "
                                "If revision is not specified, it is assumed latest one.")

    args = parser.parse_args(*args)
    # parameter validation
    validate_common_graph_args(args)

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=None) if args.path else None

    # Basic collaborators, remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial,
                                               overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)

    if path:
        deps_graph = conan_api.graph.load_graph_consumer(path, args.name, args.version,
                                                         args.user, args.channel,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.update,
                                                         check_updates=args.check_updates,
                                                         is_build_require=args.build_require)
    else:
        deps_graph = conan_api.graph.load_graph_requires(args.requires, args.tool_requires,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.update,
                                                         check_updates=args.check_updates)
    print_graph_basic(deps_graph)
    deps_graph.report_graph_error()
    conan_api.graph.analyze_binaries(deps_graph, args.build, remotes=remotes, update=args.update,
                                     lockfile=lockfile)
    print_graph_packages(deps_graph)

    ConanOutput().title("Retrieving and computing closest binaries")
    # compute ref and conaninfo of the first missing binary
    ref, conaninfo = conan_api.graph.find_first_missing_binary(deps_graph, args.missing)
    pkglist = conan_api.list.explain_missing_binaries(ref, conaninfo, remotes)

    ConanOutput().title("Closest binaries")
    return {"closest_binaries": pkglist.serialize()}


def outdated_text_formatter(result):
    cli_out_write("======== Outdated dependencies ========", fg=Color.BRIGHT_MAGENTA)

    if len(result) == 0:
        cli_out_write("No outdated dependencies in graph", fg=Color.BRIGHT_YELLOW)

    for key, value in result.items():
        current_versions_set = list({str(v) for v in value["cache_refs"]})
        cli_out_write(key, fg=Color.BRIGHT_YELLOW)
        cli_out_write(
            f'    Current versions:  {", ".join(current_versions_set) if value["cache_refs"] else "No version found in cache"}', fg=Color.BRIGHT_CYAN)
        cli_out_write(
            f'    Latest in remote(s):  {value["latest_remote"]["ref"]} - {value["latest_remote"]["remote"]}',
            fg=Color.BRIGHT_CYAN)
        if value["version_ranges"]:
            cli_out_write(f'    Version ranges: ' + str(value["version_ranges"])[1:-1], fg=Color.BRIGHT_CYAN)


def outdated_json_formatter(result):
    output = {key: {"current_versions": list({str(v) for v in value["cache_refs"]}),
                    "version_ranges": [str(r) for r in value["version_ranges"]],
                    "latest_remote": [] if value["latest_remote"] is None
                                        else {"ref": str(value["latest_remote"]["ref"]),
                                              "remote": str(value["latest_remote"]["remote"])}}
              for key, value in result.items()}
    cli_out_write(json.dumps(output))


@conan_subcommand(formatters={"text": outdated_text_formatter, "json": outdated_json_formatter})
def graph_outdated(conan_api, parser, subparser, *args):
    """
    List the dependencies in the graph and it's newer versions in the remote
    """
    common_graph_args(subparser)
    subparser.add_argument("--check-updates", default=False, action="store_true",
                           help="Check if there are recipe updates")
    subparser.add_argument("--build-require", action='store_true', default=False,
                           help='Whether the provided reference is a build-require')
    args = parser.parse_args(*args)
    # parameter validation
    validate_common_graph_args(args)
    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=None) if args.path else None

    # Basic collaborators, remotes, lockfile, profiles
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial,
                                               overrides=overrides)
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)

    if path:
        deps_graph = conan_api.graph.load_graph_consumer(path, args.name, args.version,
                                                         args.user, args.channel,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.update,
                                                         check_updates=args.check_updates,
                                                         is_build_require=args.build_require)
    else:
        deps_graph = conan_api.graph.load_graph_requires(args.requires, args.tool_requires,
                                                         profile_host, profile_build, lockfile,
                                                         remotes, args.update,
                                                         check_updates=args.check_updates)
    print_graph_basic(deps_graph)

    # Data structure to store info per library
    # DO NOT USE this API call yet, it is not stable
    outdated = conan_api.list.outdated(deps_graph, remotes)
    return outdated
