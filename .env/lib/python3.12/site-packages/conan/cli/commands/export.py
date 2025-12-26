import json
import os

from conan.api.model import MultiPackagesList, PackagesList
from conan.api.output import cli_out_write
from conan.cli.command import conan_command, OnceArgument
from conan.cli.args import add_reference_args


def common_args_export(parser):
    parser.add_argument("path", help="Path to a folder containing a recipe (conanfile.py). "
                                     "Defaults to current directory",
                        default=".", nargs="?")
    add_reference_args(parser)


def json_export(data):
    cli_out_write(json.dumps({"reference": data["reference"].repr_notime()}))


def pkglist_export(data):
    cli_out_write(json.dumps(data["pkglist"], indent=4))


@conan_command(group="Creator", formatters={"json": json_export, "pkglist": pkglist_export})
def export(conan_api, parser, *args):
    """
    Export a recipe to the Conan package cache.
    """
    common_args_export(parser)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--remote", action="append", default=None,
                       help='Look in the specified remote or remotes server')
    group.add_argument("-nr", "--no-remote", action="store_true",
                       help='Do not use remote, resolve exclusively in the cache')
    parser.add_argument("-l", "--lockfile", action=OnceArgument,
                        help="Path to a lockfile.")
    parser.add_argument("--lockfile-out", action=OnceArgument,
                        help="Filename of the updated lockfile")
    parser.add_argument("--lockfile-partial", action="store_true",
                        help="Do not raise an error if some dependency is not found in lockfile")
    parser.add_argument("--build-require", action='store_true', default=False,
                        help='Whether the provided reference is a build-require')
    args = parser.parse_args(*args)

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=True)
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial)
    ref, conanfile = conan_api.export.export(path=path,
                                             name=args.name, version=args.version,
                                             user=args.user, channel=args.channel,
                                             lockfile=lockfile,
                                             remotes=remotes)
    lockfile = conan_api.lockfile.update_lockfile_export(lockfile, conanfile, ref,
                                                         args.build_require)
    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, cwd)

    exported_list = PackagesList()
    exported_list.add_ref(ref)

    pkglist = MultiPackagesList()
    pkglist.add("Local Cache", exported_list)

    return {
        "pkglist": pkglist.serialize(),
        "reference": ref
    }
