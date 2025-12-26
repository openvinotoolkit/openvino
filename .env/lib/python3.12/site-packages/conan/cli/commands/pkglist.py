from conan.api.conan_api import ConanAPI
from conan.api.model import MultiPackagesList
from conan.cli import make_abs_path
from conan.cli.command import conan_command, conan_subcommand
from conan.cli.commands.list import print_list_text, print_list_json
from conan.cli.formatters.list import list_packages_html


@conan_command(group="Consumer")
def pkglist(conan_api: ConanAPI, parser, *args):  # noqa
    """
    Several operations over package lists
    """


@conan_subcommand(formatters={"text": print_list_text,
                              "json": print_list_json,
                              "html": list_packages_html})
def pkglist_find_remote(conan_api, parser, subparser, *args):
    """
    (Experimental) Find the remotes of a list of packages in the cache
    """
    subparser.add_argument('list', help="Input package list")
    subparser.add_argument("-r", "--remote", default=None, action="append",
                           help="Remote names. Accepts wildcards "
                                "('*' means all the remotes available)")
    args = parser.parse_args(*args)

    listfile = make_abs_path(args.list)
    multi_pkglist = MultiPackagesList.load(listfile)
    package_list = multi_pkglist["Local Cache"]
    selected_remotes = conan_api.remotes.list(args.remote)
    result = conan_api.list.find_remotes(package_list, selected_remotes)
    return {
        "results": result.serialize(),
        "conan_api": conan_api,
        "cli_args": " ".join([f"{arg}={getattr(args, arg)}"
                              for arg in vars(args) if getattr(args, arg)])
    }


@conan_subcommand(formatters={"text": print_list_text,
                              "json": print_list_json,
                              "html": list_packages_html})
def pkglist_merge(conan_api, parser, subparser, *args):
    """
    (Experimental) Merge several package lists into a single one
    """
    subparser.add_argument("-l", "--list", help="Package list file", action="append")
    args = parser.parse_args(*args)

    result = MultiPackagesList()
    for pkg_list in args.list:
        listfile = make_abs_path(pkg_list)
        multi_pkglist = MultiPackagesList.load(listfile)
        result.merge(multi_pkglist)

    return {
        "results": result.serialize(),
        "conan_api": conan_api,
        "cli_args": " ".join([f"{arg}={getattr(args, arg)}"
                              for arg in vars(args) if getattr(args, arg)])
    }
