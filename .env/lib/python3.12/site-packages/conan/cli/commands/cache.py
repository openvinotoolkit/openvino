import json

from conan.api.conan_api import ConanAPI
from conan.api.model import ListPattern, MultiPackagesList
from conan.api.output import cli_out_write, ConanOutput
from conan.cli import make_abs_path
from conan.cli.command import conan_command, conan_subcommand, OnceArgument
from conan.cli.commands.list import print_list_text, print_list_json
from conan.errors import ConanException
from conan.api.model import PkgReference
from conan.api.model import RecipeReference


def json_export(data):
    cli_out_write(json.dumps({"cache_path": data}))


@conan_command(group="Consumer")
def cache(conan_api: ConanAPI, parser, *args):
    """
    Perform file operations in the local cache (of recipes and/or packages).
    """
    pass


@conan_subcommand(formatters={"text": cli_out_write, "json": json_export})
def cache_path(conan_api: ConanAPI, parser, subparser, *args):
    """
    Show the path to the Conan cache for a given reference.
    """
    subparser.add_argument("reference", help="Recipe reference or Package reference")
    subparser.add_argument("--folder", choices=['export_source', 'source', 'build', 'metadata'],
                           help="Path to show. The 'build' requires a package reference. "
                                "If the argument is not passed, it shows 'exports' path for recipe references "
                                "and 'package' folder for package references.")

    args = parser.parse_args(*args)
    try:
        pref = PkgReference.loads(args.reference)
    except ConanException:
        pref = None

    if not pref:  # Not a package reference
        ref = RecipeReference.loads(args.reference)
        if args.folder is None:
            path = conan_api.cache.export_path(ref)
        elif args.folder == "export_source":
            path = conan_api.cache.export_source_path(ref)
        elif args.folder == "source":
            path = conan_api.cache.source_path(ref)
        elif args.folder == "metadata":
            path = conan_api.cache.recipe_metadata_path(ref)
        else:
            raise ConanException(f"'--folder {args.folder}' requires a valid package reference")
    else:
        if args.folder is None:
            path = conan_api.cache.package_path(pref)
        elif args.folder == "build":
            path = conan_api.cache.build_path(pref)
        elif args.folder == "metadata":
            path = conan_api.cache.package_metadata_path(pref)
        else:
            raise ConanException(f"'--folder {args.folder}' requires a recipe reference")
    return path


@conan_subcommand(formatters={"text": cli_out_write})
def cache_ref(conan_api: ConanAPI, parser, subparser, *args):
    """
    Show the reference for a given Conan cache folder
    """
    subparser.add_argument("path", help="Path to a Conan cache folder")
    args = parser.parse_args(*args)
    ref = conan_api.cache.path_to_ref(args.path)
    if ref is None:
        raise ConanException("Reference for this path not found in cache")
    return ref.repr_notime()


@conan_subcommand()
def cache_clean(conan_api: ConanAPI, parser, subparser, *args):
    """
    Remove non-critical folders from the cache, like source, build and/or download
    (.tgz store) ones.
    """
    subparser.add_argument("pattern", nargs="?", help="Selection pattern for references to clean")
    subparser.add_argument("-l", "--list", action=OnceArgument,
                            help="Package list of packages to clean")
    subparser.add_argument("-s", "--source", action='store_true', default=False,
                           help="Clean source folders")
    subparser.add_argument("-b", "--build", action='store_true', default=False,
                           help="Clean build folders")
    subparser.add_argument("-d", "--download", action='store_true', default=False,
                           help="Clean download and metadata folders")
    subparser.add_argument("-t", "--temp", action='store_true', default=False,
                           help="Clean temporary folders")
    subparser.add_argument("-bs", "--backup-sources", action='store_true', default=False,
                           help="Clean backup sources")
    subparser.add_argument('-p', '--package-query', action=OnceArgument,
                           help="Remove only the packages matching a specific query, e.g., "
                                "os=Windows AND (arch=x86 OR compiler=gcc)")
    args = parser.parse_args(*args)

    # If pattern is None, it will be replaced by "*"
    if args.pattern and args.list:
        raise ConanException("Cannot specify both pattern and list")

    if args.list:
        listfile = make_abs_path(args.list)
        multi_package_list = MultiPackagesList.load(listfile)
        package_list = multi_package_list["Local Cache"]
    else:
        ref_pattern = ListPattern(args.pattern or "*", rrev="*", package_id="*", prev="*")
        package_list = conan_api.list.select(ref_pattern, package_query=args.package_query)
    if args.build or args.source or args.download or args.temp or args.backup_sources:
        conan_api.cache.clean(package_list, source=args.source, build=args.build,
                              download=args.download, temp=args.temp,
                              backup_sources=args.backup_sources)
    else:
        conan_api.cache.clean(package_list)


def print_list_check_integrity_json(data):
    results = data["results"]
    myjson = json.dumps(results, indent=4)
    cli_out_write(myjson)

@conan_subcommand(formatters={"text": lambda _: (),
                              "json": print_list_check_integrity_json})
def cache_check_integrity(conan_api: ConanAPI, parser, subparser, *args):
    """
    Check the integrity of the local cache for the given references
    """
    subparser.add_argument("pattern", nargs="?",
                           help="Selection pattern for references to check integrity for")
    subparser.add_argument("-l", "--list", action=OnceArgument,
                           help="Package list of packages to check integrity for")
    subparser.add_argument('-p', '--package-query', action=OnceArgument,
                           help="Only the packages matching a specific query, e.g., "
                                "os=Windows AND (arch=x86 OR compiler=gcc)")
    args = parser.parse_args(*args)

    if args.pattern is None and args.list is None:
        raise ConanException("Missing pattern or package list file")
    if args.pattern and args.list:
        raise ConanException("Cannot specify both pattern and list")

    if args.list:
        listfile = make_abs_path(args.list)
        multi_package_list = MultiPackagesList.load(listfile)
        package_list = multi_package_list["Local Cache"]
    else:
        ref_pattern = ListPattern(args.pattern, rrev="*", package_id="*", prev="*")
        package_list = conan_api.list.select(ref_pattern, package_query=args.package_query)

    corrupted_artifacts = conan_api.cache.check_integrity(package_list, return_pkg_list=True)
    return {"results": {"Local Cache": corrupted_artifacts.serialize()},
            "conan_error": "There are corrupted artifacts, check the error logs" if corrupted_artifacts else ""}


@conan_subcommand(formatters={"text": print_list_text,
                              "json": print_list_json})
def cache_save(conan_api: ConanAPI, parser, subparser, *args):
    """
    Get the artifacts from a package list and archive them
    """
    subparser.add_argument('pattern', nargs="?",
                           help="A pattern in the form 'pkg/version#revision:package_id#revision', "
                                "e.g: zlib/1.2.13:* means all binaries for zlib/1.2.13. "
                                "If revision is not specified, it is assumed latest one.")
    subparser.add_argument("-l", "--list", help="Package list of packages to save")
    subparser.add_argument('--file', help="Save to this tgz file")
    subparser.add_argument("--no-source", action="store_true", help="Exclude the sources")
    args = parser.parse_args(*args)

    if args.pattern is None and args.list is None:
        raise ConanException("Missing pattern or package list file")
    if args.pattern and args.list:
        raise ConanException("Cannot define both the pattern and the package list file")

    if args.list:
        listfile = make_abs_path(args.list)
        multi_package_list = MultiPackagesList.load(listfile)
        package_list = multi_package_list["Local Cache"]
    else:
        ref_pattern = ListPattern(args.pattern)
        package_list = conan_api.list.select(ref_pattern)
    tgz_path = make_abs_path(args.file or "conan_cache_save.tgz")
    conan_api.cache.save(package_list, tgz_path, args.no_source)
    return {"results": {"Local Cache": package_list.serialize()}}


@conan_subcommand(formatters={"text": print_list_text,
                              "json": print_list_json})
def cache_restore(conan_api: ConanAPI, parser, subparser, *args):
    """
    Put  the artifacts from an archive into the cache
    """
    subparser.add_argument("file", help="Path to archive to restore")
    args = parser.parse_args(*args)
    path = make_abs_path(args.file)
    package_list = conan_api.cache.restore(path)
    return {"results": {"Local Cache": package_list.serialize()}}


@conan_subcommand()
def cache_backup_upload(conan_api: ConanAPI, parser, subparser, *args):
    """
    Upload all the source backups present in the cache
    """
    parser.parse_args(*args)
    files = conan_api.cache.get_backup_sources()
    conan_api.upload.upload_backup_sources(files)
