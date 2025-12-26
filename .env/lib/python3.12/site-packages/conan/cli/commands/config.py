import os

from conan.api.model import Remote
from conan.api.output import cli_out_write
from conan.cli import make_abs_path
from conan.cli.command import conan_command, conan_subcommand, OnceArgument
from conan.cli.formatters import default_json_formatter
from conan.errors import ConanException


@conan_command(group='Consumer')
def config(conan_api, parser, *args):  # noqa
    """
    Manage the Conan configuration in the Conan home.
    """


@conan_subcommand()
def config_install(conan_api, parser, subparser, *args):
    """
    Install the configuration (remotes, profiles, conf), from git, http or a folder, into the
    Conan home folder.
    """
    subparser.add_argument("item",
                           help="git repository, local file or folder or zip file (local or "
                                "http) where the configuration is stored")

    ssl_subgroup = subparser.add_mutually_exclusive_group()
    ssl_subgroup.add_argument("--verify-ssl", nargs="?", default="True",
                              help='Verify SSL connection when downloading file')
    ssl_subgroup.add_argument("--insecure", action="store_false", default=None,
                              help="Allow insecure server connections when using SSL. "
                                   "Equivalent to --verify-ssl=False",
                              dest="verify_ssl")
    subparser.add_argument("-t", "--type", choices=["git", "dir", "file", "url"],
                           help='Type of remote config')
    subparser.add_argument("-a", "--args",
                           help='String with extra arguments for "git clone"')
    subparser.add_argument("-sf", "--source-folder",
                           help='Install files only from a source subfolder from the '
                                'specified origin')
    subparser.add_argument("-tf", "--target-folder",
                           help='Install to that path in the conan cache')

    args = parser.parse_args(*args)

    def get_bool_from_text(value):  # TODO: deprecate this
        value = value.lower()
        if value in ["1", "yes", "y", "true"]:
            return True
        if value in ["0", "no", "n", "false"]:
            return False
        raise ConanException("Unrecognized boolean value '%s'" % value)
    verify_ssl = args.verify_ssl if isinstance(args.verify_ssl, bool) \
        else get_bool_from_text(args.verify_ssl)
    conan_api.config.install(args.item, verify_ssl, args.type, args.args,
                             source_folder=args.source_folder,
                             target_folder=args.target_folder)


@conan_subcommand()
def config_install_pkg(conan_api, parser, subparser, *args):
    """
    (Experimental) Install the configuration (remotes, profiles, conf), from a Conan package
    or from a conanconfig.yml file
    """
    subparser.add_argument("reference", nargs="?",
                           help="Package reference 'pkg/version' to install configuration from "
                                "or path to 'conanconfig.yml' file")
    subparser.add_argument("-l", "--lockfile", action=OnceArgument,
                           help="Path to a lockfile. Use --lockfile=\"\" to avoid automatic use of "
                                "existing 'conan.lock' file")
    subparser.add_argument("--lockfile-partial", action="store_true",
                           help="Do not raise an error if some dependency is not found in lockfile")
    subparser.add_argument("--lockfile-out", action=OnceArgument,
                           help="Filename of the updated lockfile")
    subparser.add_argument("-f", "--force", action='store_true',
                           help="Force the re-installation of configuration")
    subparser.add_argument("--url", action=OnceArgument,
                           help="(Experimental) Provide Conan repository URL "
                                "(for first install without remotes)")
    subparser.add_argument("-pr", "--profile", help="Profile to install config")
    subparser.add_argument("-s", "--settings", action="append", help="Settings to install config")
    subparser.add_argument("-o", "--options", action="append", help="Options to install config")
    args = parser.parse_args(*args)

    path = make_abs_path(args.reference or ".")
    if os.path.isdir(path):
        path = os.path.join(path, "conanconfig.yml")
    path = path if os.path.exists(path) else None
    if path is None and args.reference is None:
        raise ConanException("Must provide a package reference or a path to a conanconfig.yml file")

    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               partial=args.lockfile_partial)

    try:
        default_profile = args.profile or conan_api.profiles.get_default_build()
    except ConanException:  # it can fail if the default profile doesn't exist yet
        default_profile = None
    profiles = [default_profile] if default_profile else []
    profile = conan_api.profiles.get_profile(profiles, args.settings, args.options)
    remotes = [Remote("config_install_url", url=args.url)] if args.url else None

    if path:
        conanconfig = path
        refs = conan_api.config.install_conanconfig(conanconfig, lockfile=lockfile, force=args.force,
                                                    remotes=remotes, profile=profile)
    else:
        refs = conan_api.config.install_package(args.reference, lockfile=lockfile, force=args.force,
                                                remotes=remotes, profile=profile)
    lockfile = conan_api.lockfile.add_lockfile(lockfile, config_requires=refs)
    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out)


def _list_text_formatter(confs):
    for k, v in confs.items():
        cli_out_write(f"{k}: {v}")


@conan_subcommand(formatters={"text": cli_out_write})
def config_home(conan_api, parser, subparser, *args):  # noqa
    """
    Show the Conan home folder.
    """
    parser.parse_args(*args)
    return conan_api.config.home()


@conan_subcommand(formatters={"text": _list_text_formatter, "json": default_json_formatter})
def config_list(conan_api, parser, subparser, *args):
    """
    Show all the Conan available configurations: core and tools.
    """
    subparser.add_argument('pattern', nargs="?",
                           help="Filter configuration items that matches this pattern")
    args = parser.parse_args(*args)
    confs = conan_api.config.conf_list()
    if args.pattern:
        p = args.pattern.lower()
        confs = {k: v for k, v in confs.items() if p in k.lower() or p in v.lower()}
    return confs


@conan_subcommand(formatters={"text": _list_text_formatter, "json": default_json_formatter})
def config_show(conan_api, parser, subparser, *args):
    """
    Get the value of the specified conf
    """
    subparser.add_argument('pattern', help='Conf item(s) pattern for which to query their value')
    args = parser.parse_args(*args)

    return conan_api.config.show(args.pattern)


@conan_subcommand()
def config_clean(conan_api, parser, subparser, *args):  # noqa
    """
    (Experimental) Clean the configuration files in the Conan home folder, while keeping
    installed packages
    """
    parser.parse_args(*args)
    conan_api.config.clean()
