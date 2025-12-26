import os
import shutil

from conan.api.output import ConanOutput
from conan.cli.args import add_lockfile_args, add_common_install_arguments
from conan.cli.command import conan_command, OnceArgument
from conan.cli.commands.export import common_args_export
from conan.cli.formatters.graph import format_graph_json
from conan.cli.printers import print_profiles
from conan.cli.printers.graph import print_graph_packages, print_graph_basic
from conan.errors import ConanException


@conan_command(group="Creator", formatters={"json": format_graph_json})
def create(conan_api, parser, *args):
    """
    Create a package.
    """
    common_args_export(parser)
    add_lockfile_args(parser)
    add_common_install_arguments(parser)
    parser.add_argument("--build-require", action='store_true', default=False,
                        help='Whether the package being created is a build-require (to be used'
                             ' as tool_requires() by other packages)')
    parser.add_argument("-tf", "--test-folder", action=OnceArgument,
                        help='Alternative test folder name. By default it is "test_package". '
                             'Use "" to skip the test stage')
    parser.add_argument("-tm", "--test-missing", action='store_true', default=False,
                        help='Run the test_package checks only if the package is built from source'
                             ' but not if it already existed (using --build=missing)')
    parser.add_argument("-bt", "--build-test", action="append",
                        help="Same as '--build' but only for the test_package requires. By default"
                             " if not specified it will take the '--build' value if specified")
    raw_args = args[0]
    args = parser.parse_args(*args)

    if args.test_missing and args.test_folder == "":
        raise ConanException('--test-folder="" is incompatible with --test-missing')

    cwd = os.getcwd()
    path = conan_api.local.get_conanfile_path(args.path, cwd, py=True)
    overrides = eval(args.lockfile_overrides) if args.lockfile_overrides else None
    lockfile = conan_api.lockfile.get_lockfile(lockfile=args.lockfile,
                                               conanfile_path=path,
                                               cwd=cwd,
                                               partial=args.lockfile_partial,
                                               overrides=overrides)
    remotes = conan_api.remotes.list(args.remote) if not args.no_remote else []
    profile_host, profile_build = conan_api.profiles.get_profiles_from_args(args)

    ref, conanfile = conan_api.export.export(path=path,
                                             name=args.name, version=args.version,
                                             user=args.user, channel=args.channel,
                                             lockfile=lockfile,
                                             remotes=remotes)

    # FIXME: Dirty: package type still raw, not processed yet
    is_build = args.build_require or conanfile.package_type == "build-scripts"
    # The package_type is not fully processed at export
    is_python_require = conanfile.package_type == "python-require"
    lockfile = conan_api.lockfile.update_lockfile_export(lockfile, conanfile, ref, is_build)

    print_profiles(profile_host, profile_build)
    runner = conan_api.command.get_runner(profile_host)
    if runner is not None:
        return runner(conan_api, 'create', profile_host, profile_build, args, raw_args).run()

    if args.build is not None and args.build_test is None:
        args.build_test = args.build

    install_error = None
    if is_python_require:
        deps_graph = conan_api.graph.load_graph_requires([], [],
                                                         profile_host=profile_host,
                                                         profile_build=profile_build,
                                                         lockfile=lockfile,
                                                         remotes=remotes, update=args.update,
                                                         python_requires=[ref])
    else:
        requires = [ref] if not is_build else None
        tool_requires = [ref] if is_build else None
        if conanfile.vendor:  # Automatically allow repackaging for conan create
            pr = profile_build if is_build else profile_host
            pr.conf.update("&:tools.graph:vendor", "build")
        deps_graph = conan_api.graph.load_graph_requires(requires, tool_requires,
                                                         profile_host=profile_host,
                                                         profile_build=profile_build,
                                                         lockfile=lockfile,
                                                         remotes=remotes, update=args.update)
        print_graph_basic(deps_graph)
        deps_graph.report_graph_error()

        # Not specified, force build the tested library
        build_modes = [ref.repr_notime()] if args.build is None else args.build
        if args.build is None and conanfile.build_policy == "never":
            raise ConanException(
                "This package cannot be created, 'build_policy=never', it can only be 'export-pkg'")
        conan_api.graph.analyze_binaries(deps_graph, build_modes, remotes=remotes,
                                         update=args.update, lockfile=lockfile)
        print_graph_packages(deps_graph)

        install_error = conan_api.install.install_binaries(deps_graph=deps_graph, remotes=remotes,
                                                           return_install_error=True)

        # We update the lockfile, so it will be updated for later ``test_package``
        lockfile = conan_api.lockfile.update_lockfile(lockfile, deps_graph, args.lockfile_packages,
                                                      clean=args.lockfile_clean)

    test_package_folder = getattr(conanfile, "test_package_folder", None) \
        if args.test_folder is None else args.test_folder
    test_conanfile_path = _get_test_conanfile_path(test_package_folder, path)
    # If the user provide --test-missing and the binary was not built from source, skip test_package
    if args.test_missing and deps_graph.root.edges\
            and deps_graph.root.edges[0].dst.binary != "Build":
        test_conanfile_path = None  # disable it

    if test_conanfile_path and not install_error:
        # TODO: We need arguments for:
        #  - decide update policy "--test_package_update"
        # If it is a string, it will be injected always, if it is a RecipeReference, then it will
        # be replaced only if ``python_requires = "tested_reference_str"``
        tested_python_requires = ref.repr_notime() if is_python_require else ref
        from conan.cli.commands.test import run_test
        # The test_package do not make the "conan create" command return a different graph or
        # produce a different lockfile. The result is always the same, irrespective of test_package
        run_test(conan_api, test_conanfile_path, ref, profile_host, profile_build, remotes, lockfile,
                 update=None, build_modes=args.build, build_modes_test=args.build_test,
                 tested_python_requires=tested_python_requires, tested_graph=deps_graph)

    conan_api.lockfile.save_lockfile(lockfile, args.lockfile_out, cwd)
    return {"graph": deps_graph,
            "conan_api": conan_api,
            "conan_error": install_error}


def _check_tested_reference_matches(deps_graph, tested_ref, out):
    """ Check the test_profile_override_conflict test. If we are testing a build require
    but we specify the build require with a different version in the profile, it has priority,
    it is correct but weird and likely a mistake"""
    # https://github.com/conan-io/conan/issues/10453
    direct_refs = [n.conanfile.ref for n in deps_graph.root.neighbors()]
    # There is a reference with same name but different
    missmatch = [ref for ref in direct_refs if ref.name == tested_ref.name and ref != tested_ref]
    if missmatch:
        out.warning("The package created was '{}' but the reference being "
                    "tested is '{}'".format(missmatch[0], tested_ref))


def test_package(conan_api, deps_graph, test_conanfile_path):
    out = ConanOutput()
    out.title("Testing the package")
    # TODO: Better modeling when we are testing a python_requires
    conanfile = deps_graph.root.conanfile
    if len(deps_graph.nodes) == 1 and not hasattr(conanfile, "python_requires"):
        raise ConanException("The conanfile at '{}' doesn't declare any requirement, "
                             "use `self.tested_reference_str` to require the "
                             "package being created.".format(test_conanfile_path))
    conanfile_folder = os.path.dirname(test_conanfile_path)
    # To make sure the folders are correct
    conanfile.folders.set_base_folders(conanfile_folder, output_folder=None)
    if conanfile.build_folder and conanfile.build_folder != conanfile.source_folder:
        # should be the same as build folder, but we can remove it
        out.info("Removing previously existing 'test_package' build folder: "
                 f"{conanfile.build_folder}")
        shutil.rmtree(conanfile.build_folder, ignore_errors=True)
        os.makedirs(conanfile.build_folder, exist_ok=True)
    conanfile.output.info(f"Test package build: {conanfile.folders.build}")
    conanfile.output.info(f"Test package build folder: {conanfile.build_folder}")
    conan_api.install.install_consumer(deps_graph=deps_graph,
                                       source_folder=conanfile_folder)

    out.title("Testing the package: Building")
    conan_api.local.build(conanfile)

    out.title("Testing the package: Executing test")
    conanfile.output.highlight("Running test()")
    conan_api.local.test(conanfile)


def _get_test_conanfile_path(tf, conanfile_path):
    """Searches in the declared test_folder or in the standard "test_package"
    """
    if tf == "":  # Now if parameter --test-folder="" we have to skip tests
        return None
    base_folder = os.path.dirname(conanfile_path)
    test_conanfile_path = os.path.join(base_folder, tf or "test_package", "conanfile.py")
    if os.path.exists(test_conanfile_path):
        return test_conanfile_path
    elif tf:
        raise ConanException(f"test folder '{tf}' not available, or it doesn't have a conanfile.py")
