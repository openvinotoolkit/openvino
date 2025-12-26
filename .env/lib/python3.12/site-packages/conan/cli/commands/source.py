import os

from conan.cli.command import conan_command
from conan.cli.args import add_reference_args


@conan_command(group="Creator")
def source(conan_api, parser, *args):
    """
    Call the source() method.
    """
    parser.add_argument("path", help="Path to a folder containing a conanfile.py. "
                                     "Defaults to current directory",
                        default=".", nargs="?")
    add_reference_args(parser)
    args = parser.parse_args(*args)

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=True)
    enabled_remotes = conan_api.remotes.list()  # for python_requires not local
    # TODO: Missing lockfile for python_requires
    conan_api.local.source(path, name=args.name, version=args.version, user=args.user,
                           channel=args.channel, remotes=enabled_remotes)
