from conan.api.conan_api import ConanAPI
from conan.api.model import ListPattern, MultiPackagesList
from conan.api.output import ConanOutput
from conan.cli import make_abs_path
from conan.cli.command import conan_command, OnceArgument
from conan.cli.commands.list import print_list_text, print_list_json
from conan.errors import ConanException


@conan_command(group="Creator", formatters={"text": print_list_text,
                                            "json": print_list_json})
def download(conan_api: ConanAPI, parser, *args):
    """
    Download (without installing) a single conan package from a remote server.

    It downloads just the package, but not its transitive dependencies, and it will not call
    any generate, generators or deployers.
    It can download multiple packages if patterns are used, and also works with
    queries over the package binaries.
    """

    parser.add_argument('pattern', nargs="?",
                        help="A pattern in the form 'pkg/version#revision:package_id#revision', "
                             "e.g: \"zlib/1.2.13:*\" means all binaries for zlib/1.2.13. "
                             "If revision is not specified, it is assumed latest one.")
    parser.add_argument("--only-recipe", action='store_true', default=False,
                        help='Download only the recipe/s, not the binary packages.')
    parser.add_argument('-p', '--package-query', default=None, action=OnceArgument,
                        help="Only download packages matching a specific query. e.g: os=Windows AND "
                             "(arch=x86 OR compiler=gcc)")
    parser.add_argument("-r", "--remote", action=OnceArgument, required=True,
                        help='Download from this specific remote')
    parser.add_argument("-m", "--metadata", action='append',
                        help='Download the metadata matching the pattern, even if the package is '
                             'already in the cache and not downloaded')
    parser.add_argument("-l", "--list", help="Package list file")

    args = parser.parse_args(*args)
    if args.pattern is None and args.list is None:
        raise ConanException("Missing pattern or package list file")
    if args.pattern and args.list:
        raise ConanException("Cannot define both the pattern and the package list file")

    remote = conan_api.remotes.get(args.remote)

    if args.list:
        listfile = make_abs_path(args.list)
        multi_package_list = MultiPackagesList.load(listfile)
        try:
            package_list = multi_package_list[remote.name]
        except KeyError:
            raise ConanException(f"The current package list does not contain remote '{remote.name}'")
        if args.only_recipe:
            package_list.only_recipes()
    else:
        ref_pattern = ListPattern(args.pattern, package_id="*", only_recipe=args.only_recipe)
        package_list = conan_api.list.select(ref_pattern, args.package_query, remote)

    if package_list:
        conan_api.download.download_full(package_list, remote, args.metadata)
    else:
        ConanOutput().warning(f"No packages were downloaded because the package list is empty.")

    return {"results": {"Local Cache": package_list.serialize()}}
