from collections import OrderedDict

from conan.api.conan_api import ConanAPI
from conan.api.model import ListPattern
from conan.cli.command import conan_command
from conan.cli.commands.list import print_list_text, print_list_json
from conan.errors import ConanException


@conan_command(group="Consumer", formatters={"text": print_list_text,
                                             "json": print_list_json})
def search(conan_api: ConanAPI, parser, *args):
    """
    Search for package recipes in all the remotes (by default), or a remote.
    """
    parser.add_argument('reference', help="Recipe reference to search for. "
                                          "It can contain * as wildcard at any reference field.")
    parser.add_argument("-r", "--remote", action="append",
                        help="Remote names. Accepts wildcards. If not specified it searches "
                             "in all the remotes")
    args = parser.parse_args(*args)
    ref_pattern = ListPattern(args.reference, rrev=None)
    if ref_pattern.package_id is not None or ref_pattern.rrev is not None:
        raise ConanException("Specifying a recipe revision or package ID is not allowed")

    remotes = conan_api.remotes.list(args.remote)
    if not remotes:
        raise ConanException("There are no remotes to search from")

    results = OrderedDict()
    for remote in remotes:
        try:
            pkglist = conan_api.list.select(ref_pattern, package_query=None, remote=remote)
        except Exception as e:
            results[remote.name] = {"error": str(e)}
        else:
            results[remote.name] = pkglist.serialize()
    return {
        "results": results
    }
