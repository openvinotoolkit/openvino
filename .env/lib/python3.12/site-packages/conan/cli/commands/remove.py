from conan.api.conan_api import ConanAPI
from conan.api.model import ListPattern, MultiPackagesList, PackagesList
from conan.api.output import cli_out_write, ConanOutput
from conan.api.input import UserInput
from conan.cli import make_abs_path
from conan.cli.command import conan_command, OnceArgument
from conan.cli.commands.list import print_list_json, print_serial
from conan.errors import ConanException


def summary_remove_list(results):
    """ Do a little format modification to serialized
    package list so it looks prettier on text output
    """
    cli_out_write("Remove summary:")
    info = results["results"]
    result = {}
    for remote, remote_info in info.items():
        new_info = result.setdefault(remote, {})
        for ref, content in remote_info.items():
            for rev, rev_content in content.get("revisions", {}).items():
                pkgids = rev_content.get('packages')
                if pkgids is None:
                    new_info.setdefault(f"{ref}#{rev}", "Removed recipe and all binaries")
                else:
                    new_info.setdefault(f"{ref}#{rev}", f"Removed binaries: {list(pkgids)}")
    print_serial(result)


@conan_command(group="Consumer", formatters={"text": summary_remove_list,
                                             "json": print_list_json})
def remove(conan_api: ConanAPI, parser, *args):
    """
    Remove recipes or packages from local cache or a remote.

    - If no remote is specified (-r), the removal will be done in the local conan cache.
    - If a recipe reference is specified, it will remove the recipe and all the packages, unless -p
      is specified, in that case, only the packages matching the specified query (and not the recipe)
      will be removed.
    - If a package reference is specified, it will remove only the package.
    """
    parser.add_argument('pattern', nargs="?",
                        help="A pattern in the form 'pkg/version#revision:package_id#revision', "
                             "e.g: \"zlib/1.2.13:*\" means all binaries for zlib/1.2.13. "
                             "If revision is not specified, it is assumed latest one.")
    parser.add_argument('-c', '--confirm', default=False, action='store_true',
                        help='Remove without requesting a confirmation')
    parser.add_argument('-p', '--package-query', action=OnceArgument,
                        help="Remove all packages (empty) or provide a query: "
                             "os=Windows AND (arch=x86 OR compiler=gcc)")
    parser.add_argument('-r', '--remote', action=OnceArgument,
                        help='Will remove from the specified remote')
    parser.add_argument("-l", "--list", help="Package list file")
    parser.add_argument('--lru', default=None, action=OnceArgument,
                        help="Remove recipes and binaries that have not been recently used. Use a"
                             " time limit like --lru=5d (days) or --lru=4w (weeks),"
                             " h (hours), m(minutes)")
    parser.add_argument("--dry-run", default=False, action="store_true",
                        help="Do not remove any items, only print those which would be removed")
    args = parser.parse_args(*args)

    if args.pattern is None and args.list is None:
        raise ConanException("Missing pattern or package list file")
    if args.pattern and args.list:
        raise ConanException("Cannot define both the pattern and the package list file")
    if args.package_query and args.list:
        raise ConanException("Cannot define package-query and the package list file")
    if args.remote and args.lru:
        raise ConanException("'--lru' cannot be used in remotes, only in cache")
    if args.list and args.lru:
        raise ConanException("'--lru' cannot be used with input package list")

    ui = UserInput(conan_api.config.get("core:non_interactive"))
    remote = conan_api.remotes.get(args.remote) if args.remote else None
    cache_name = "Local Cache" if not remote else remote.name

    def confirmation(message):
        return args.confirm or ui.request_boolean(message)

    if args.list:
        listfile = make_abs_path(args.list)
        multi_package_list = MultiPackagesList.load(listfile)
        package_list = multi_package_list[cache_name]
        refs_to_remove = list(package_list.items())
        if not refs_to_remove:  # the package list might contain only refs, no revs
            ConanOutput().warning("Nothing to remove, package list do not contain recipe revisions")
    else:
        ref_pattern = ListPattern(args.pattern, rrev="*", prev="*")
        if ref_pattern.package_id is None and args.package_query is not None:
            raise ConanException('--package-query supplied but the pattern does not match packages')
        package_list = conan_api.list.select(ref_pattern, args.package_query, remote, lru=args.lru)
        multi_package_list = MultiPackagesList()
        multi_package_list.add(cache_name, package_list)

    result = PackagesList()
    for ref, packages in package_list.items():
        ref_dict = package_list.recipe_dict(ref).copy()
        packages_dict = ref_dict.pop("packages", None)
        if packages_dict is None:
            if confirmation(f"Remove the recipe and all the packages of '{ref.repr_notime()}'?"):
                if not args.dry_run:
                    conan_api.remove.recipe(ref, remote=remote)
                result.add_ref(ref)
                result.recipe_dict(ref).update(ref_dict)  # it doesn't contain "packages"
        else:
            if not packages: # weird, there is inner package-ids but without prevs
                ConanOutput().info(f"No binaries to remove for '{ref.repr_notime()}'")
                continue
            for pref, pkg_id_info in packages.items():
                if confirmation(f"Remove the package '{pref.repr_notime()}'?"):
                    if not args.dry_run:
                        conan_api.remove.package(pref, remote=remote)
                    result.add_ref(ref)
                    result.recipe_dict(ref).update(ref_dict)  # it doesn't contain "packages"
                    result.add_pref(pref, pkg_id_info)
                    pkg_dict = package_list.package_dict(pref)
                    result.package_dict(pref).update(pkg_dict)
    multi_package_list.add(cache_name, result)

    return {
        "results": multi_package_list.serialize(),
        "conan_api": conan_api
    }
