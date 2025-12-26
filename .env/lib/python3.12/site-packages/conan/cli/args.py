import argparse

from conan.cli.command import OnceArgument
from conan.errors import ConanException

_help_build_policies = '''Optional, specify which packages to build from source. Combining multiple
    '--build' options on one command line is allowed.
    Possible values:

    --build=never      Disallow build for all packages, use binary packages or fail if a binary
                       package is not found, it cannot be combined with other '--build' options.
    --build=missing    Build packages from source whose binary package is not found.
    --build=cascade    Build packages from source that have at least one dependency being built from
                       source.
    --build=[pattern]  Build packages from source whose package reference matches the pattern. The
                       pattern uses 'fnmatch' style wildcards, so '--build="*"' will build everything
                       from source.
    --build=~[pattern] Excluded packages, which will not be built from the source, whose package
                       reference matches the pattern. The pattern uses 'fnmatch' style wildcards.
    --build=missing:[pattern] Build from source if a compatible binary does not exist, only for
                              packages matching pattern.
    --build=compatible:[pattern] (Experimental) Build from source if a compatible binary does not
                                 exist, and the requested package is invalid, the closest package
                                 binary following the defined compatibility policies (method and
                                 compatibility.py)
'''


def add_lockfile_args(parser):
    group = parser.add_argument_group("lockfile arguments")
    group.add_argument("-l", "--lockfile", action=OnceArgument,
                       help="Path to a lockfile. Use --lockfile=\"\" to avoid automatic use of "
                            "existing 'conan.lock' file")
    group.add_argument("--lockfile-partial", action="store_true",
                       help="Do not raise an error if some dependency is not found in lockfile")
    group.add_argument("--lockfile-out", action=OnceArgument,
                       help="Filename of the updated lockfile")
    group.add_argument("--lockfile-packages", action="store_true",
                       help=argparse.SUPPRESS)
    group.add_argument("--lockfile-clean", action="store_true",
                       help="Remove unused entries from the lockfile")
    group.add_argument("--lockfile-overrides",
                       help="Overwrite lockfile overrides")


def add_common_install_arguments(parser):
    parser.add_argument("-b", "--build", action="append", help=_help_build_policies)

    group = parser.add_argument_group("remote arguments")
    exclusive_group = group.add_mutually_exclusive_group()
    exclusive_group.add_argument("-r", "--remote", action="append", default=None,
                                 help='Look in the specified remote or remotes server')
    exclusive_group.add_argument("-nr", "--no-remote", action="store_true",
                                 help='Do not use remote, resolve exclusively in the cache')

    update_help = ("Will install newer versions and/or revisions in the local cache "
                   "for the given reference name, or all references in the graph if no argument is supplied. "
                   "When using version ranges, it will install the latest version that "
                   "satisfies the range. It will update to the "
                   "latest revision for the resolved version range.")

    group.add_argument("-u", "--update", action="append", nargs="?", help=update_help, const="*")
    add_profiles_args(parser)


def add_profiles_args(parser):
    contexts = ["build", "host"]

    group = parser.add_argument_group("profile arguments")

    # This comes from the _AppendAction code but modified to add to the contexts
    class ContextAllAction(argparse.Action):

        def __call__(self, action_parser, namespace, values, option_string=None):
            for context in contexts:
                items = getattr(namespace, self.dest + "_" + context, None)
                items = items[:] if items else []
                items.append(values)
                setattr(namespace, self.dest + "_" + context, items)

    def create_config(short, long, example=None):
        group.add_argument(f"-{short}", f"--{long}",
                           default=None,
                           action="append",
                           dest=f"{long}_host",
                           metavar=long.upper(),
                           help=f'Apply the specified {long}. '
                                f'By default, or if specifying -{short}:h (--{long}:host), it applies to the host context. '
                                f'Use -{short}:b (--{long}:build) to specify the build context, '
                                f'or -{short}:a (--{long}:all) to specify both contexts at once'
                                + ('' if not example else f". Example: {example}"))
        for context in contexts:
            group.add_argument(f"-{short}:{context[0]}", f"--{long}:{context}",
                               default=None,
                               action="append",
                               dest=f"{long}_{context}",
                               help="")

        group.add_argument(f"-{short}:a", f"--{long}:all",
                           default=None,
                           action=ContextAllAction,
                           dest=long,
                           metavar=f"{long.upper()}_ALL",
                           help="")

    create_config("pr", "profile")
    create_config("o", "options", '-o="pkg/*:with_qt=True"')
    create_config("s", "settings", '-s="compiler=gcc"')
    create_config("c", "conf", '-c="tools.cmake.cmaketoolchain:generator=Xcode"')


def add_reference_args(parser):
    group = parser.add_argument_group("reference arguments")
    group.add_argument("--name", action=OnceArgument,
                       help='Provide a package name if not specified in conanfile')
    group.add_argument("--version", action=OnceArgument,
                       help='Provide a package version if not specified in conanfile')
    group.add_argument("--user", action=OnceArgument,
                       help='Provide a user if not specified in conanfile')
    group.add_argument("--channel", action=OnceArgument,
                       help='Provide a channel if not specified in conanfile')


def common_graph_args(subparser):
    subparser.add_argument("path", nargs="?",
                           help="Path to a folder containing a recipe (conanfile.py "
                                "or conanfile.txt) or to a recipe file. e.g., "
                                "./my_project/conanfile.txt. Defaults to the current "
                                "directory when no --requires or --tool-requires is "
                                "given",
                           default=None)
    add_common_install_arguments(subparser)
    subparser.add_argument("--requires", action="append",
                           help='Directly provide requires instead of a conanfile')
    subparser.add_argument("--tool-requires", action='append',
                                                  help='Directly provide tool-requires instead of a conanfile')
    add_reference_args(subparser)
    add_lockfile_args(subparser)


def validate_common_graph_args(args):
    if args.requires and (args.name or args.version or args.user or args.channel):
        raise ConanException("Can't use --name, --version, --user or --channel arguments with "
                             "--requires")
    if args.path and (args.requires or args.tool_requires):
        raise ConanException("--requires and --tool-requires arguments are incompatible with "
                             f"[path] '{args.path}' argument")

    if not args.requires and not args.tool_requires and args.path is None:
        args.path = "."

    # graph build-order command does not define a build-require argument
    if not args.path and getattr(args, "build_require", False):
        raise ConanException("--build-require should only be used with <path> argument")
