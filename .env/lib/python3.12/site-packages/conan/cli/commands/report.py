import os

from conan.cli.formatters.report import format_diff_html, format_diff_txt, format_diff_json
from conan.api.conan_api import ConanAPI
from conan.cli.command import conan_command, conan_subcommand



@conan_command(group="Security")
def report(conan_api: ConanAPI, parser, *args):
    """
    Gets information about the recipe and its sources.
    """


@conan_subcommand(formatters={"text": format_diff_txt,
                              "json": format_diff_json,
                              "html": format_diff_html})
def report_diff(conan_api, parser, subparser, *args):
    """
    Get the difference between two recipes with their sources.
    It can be used to compare two different versions of the same recipe, or two different recipe revisions.

    Each old/new recipe can be specified by a path to a conanfile.py and a companion reference,
    or by a reference only.

    If only a reference is specified, it will be searched in the local cache,
    or downloaded from the specified remotes. If no revision is specified, the latest revision will be used.
    """

    ref_help = ("{type} reference, e.g. 'mylib/1.0'. "
                "If used on its own, it can contain a revision, which will be resolved to the latest one if not provided, "
                "but it will be ignored if a path is specified. "
                "If used with a path, it will be used to create the reference for the recipe to be compared.")

    subparser.add_argument("-op", "--old-path", help="Path to the old recipe if comparing a local recipe is desired")
    subparser.add_argument("-or", "--old-reference", help=ref_help.format(type="Old"), required=True)

    subparser.add_argument("-np", "--new-path", help="Path to the new recipe if comparing a local recipe is desired")
    subparser.add_argument("-nr", "--new-reference", help=ref_help.format(type="New"), required=True)

    subparser.add_argument("-r", "--remote", action="append", default=None,
                       help='Look in the specified remote or remotes server')

    args = parser.parse_args(*args)

    cwd = os.getcwd()
    enabled_remotes = conan_api.remotes.list(args.remote or "*")

    result = conan_api.report.diff(args.old_reference, args.new_reference, enabled_remotes,
                                   old_path=args.old_path, new_path=args.new_path, cwd=cwd)
    return {
        "conan_api": conan_api,
        "diff": result["diff"],
        "old_export_ref": result["old_export_ref"],
        "new_export_ref": result["new_export_ref"],
        "old_cache_path": result["old_cache_path"],
        "new_cache_path": result["new_cache_path"],
        "src_prefix": result["src_prefix"],
        "dst_prefix": result["dst_prefix"],
    }
