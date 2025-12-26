import os

from conan.cli.command import conan_command


@conan_command(group="Creator")
def new(conan_api, parser, *args):
    """
    Create a new example recipe and source files from a template.
    """
    parser.add_argument("template", help="Template name, "
                        "either a predefined built-in or a user-provided one. "
                        "Available built-in templates: basic, cmake_lib, cmake_exe, header_lib, "
                        "meson_lib, meson_exe, msbuild_lib, msbuild_exe, bazel_lib, bazel_exe, "
                        "autotools_lib, autotools_exe, premake_lib, premake_exe, local_recipes_index, workspace. "
                        "E.g. 'conan new cmake_lib -d name=hello -d version=0.1'. "
                        "You can define your own templates too by inputting an absolute path "
                        "as your template, or a path relative to your conan home folder."
                        )
    parser.add_argument("-d", "--define", action="append",
                        help="Define a template argument as key=value, e.g., -d name=mypkg")
    parser.add_argument("-f", "--force", action='store_true',
                        help="Overwrite file if it already exists")
    parser.add_argument("-o", "--output", help="Output folder for the generated files",
                        default=os.getcwd())

    args = parser.parse_args(*args)
    conan_api.new.save_template(args.template, args.define, args.output, args.force)
