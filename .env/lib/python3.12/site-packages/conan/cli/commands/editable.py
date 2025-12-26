import json
import os

from conan.api.output import ConanOutput, cli_out_write
from conan.cli.args import add_reference_args
from conan.cli.command import conan_command, conan_subcommand


@conan_command(group="Creator")
def editable(conan_api, parser, *args):
    """
    Allow working with a package that resides in user folder.
    """


@conan_subcommand()
def editable_add(conan_api, parser, subparser, *args):
    """
    Define the given <path> location as the package <reference>, so when this
    package is required, it is used from this <path> location instead of the cache.
    """
    subparser.add_argument('path', help='Path to the package folder in the user workspace',
                           default=".", nargs='?')
    add_reference_args(subparser)
    subparser.add_argument("-of", "--output-folder",
                           help='The root output folder for generated and build files')
    group = subparser.add_mutually_exclusive_group()
    group.add_argument("-r", "--remote", action="append", default=None,
                       help='Look in the specified remote or remotes server')
    group.add_argument("-nr", "--no-remote", action="store_true",
                       help='Do not use remote, resolve exclusively in the cache')
    args = parser.parse_args(*args)

    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    cwd = os.getcwd()
    ref = conan_api.local.editable_add(args.path, args.name, args.version, args.user, args.channel,
                                       cwd, args.output_folder, remotes=remotes)
    ConanOutput().success("Reference '{}' in editable mode".format(ref))


@conan_subcommand()
def editable_remove(conan_api, parser, subparser, *args):
    """
    Remove the "editable" mode for this reference.
    """
    subparser.add_argument("path", nargs="?",
                           help="Path to a folder containing a recipe conanfile.py "
                                "or to a recipe file. e.g., "
                                "./my_project/conanfile.py.",
                           default=None)
    subparser.add_argument("-r", "--refs", action="append",
                           help='Directly provide reference patterns')
    args = parser.parse_args(*args)
    if not args.refs and args.path is None:
        args.path = "."
    # TODO: Fix this API to use get_conanfile_path
    editables = conan_api.local.editable_remove(args.path, args.refs)
    out = ConanOutput()
    if editables:
        for ref, info in editables.items():
            out.success(f"Removed editable '{ref}': {info['path']}")
    else:
        out.warning("No editables were removed")


def print_editables_json(data):
    results = {str(k): v for k, v in data.items()}
    myjson = json.dumps(results, indent=4)
    cli_out_write(myjson)


def print_editables_text(data):
    for k, v in data.items():
        cli_out_write("%s" % k)
        cli_out_write("    Path: %s" % v["path"])
        if v.get("output_folder"):
            cli_out_write("    Output: %s" % v["output_folder"])


@conan_subcommand(formatters={"text": print_editables_text, "json": print_editables_json})
def editable_list(conan_api, parser, subparser, *args):
    """
    List all the packages in editable mode.
    """
    parser.parse_args(*args)
    editables = conan_api.local.editable_list()
    return editables
