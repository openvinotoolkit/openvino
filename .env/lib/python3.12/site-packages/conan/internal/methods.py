import os

from conan.api.output import ConanOutput
from conan.errors import ConanException
from conan.internal.errors import conanfile_exception_formatter, conanfile_remove_attr
from conan.internal.paths import CONANINFO
from conan.internal.model.manifest import FileTreeManifest
from conan.api.model import PkgReference
from conan.internal.model.pkg_type import PackageType
from conan.internal.model.requires import BuildRequirements, TestRequirements, ToolRequirements
from conan.internal.util.files import mkdir, chdir, save


def run_source_method(conanfile, hook_manager):
    mkdir(conanfile.source_folder)
    with chdir(conanfile.source_folder):
        hook_manager.execute("pre_source", conanfile=conanfile)
        if hasattr(conanfile, "source"):
            conanfile.output.highlight("Calling source() in {}".format(conanfile.source_folder))
            with conanfile_exception_formatter(conanfile, "source"):
                with conanfile_remove_attr(conanfile, ['info', 'settings', "options"], "source"):
                    conanfile.source()
        hook_manager.execute("post_source", conanfile=conanfile)


def run_build_method(conanfile, hook_manager):
    if os.path.isfile(conanfile.build_folder):
        raise ConanException(f"{conanfile}: Failed to create build folder, there is already a file "
                             f"named: {conanfile.build_folder}")
    mkdir(conanfile.build_folder)
    mkdir(conanfile.package_metadata_folder)
    with chdir(conanfile.build_folder):
        hook_manager.execute("pre_build", conanfile=conanfile)
        if hasattr(conanfile, "build"):
            conanfile.output.highlight("Calling build()")
            with conanfile_exception_formatter(conanfile, "build"):
                try:
                    conanfile.build()
                except Exception:
                    hook_manager.execute("post_build_fail", conanfile=conanfile)
                    raise
        hook_manager.execute("post_build", conanfile=conanfile)


def run_package_method(conanfile, package_id, hook_manager, ref):
    """ calls the recipe "package()" method
    - Assigns folders to conanfile.package_folder, source_folder, install_folder, build_folder
    - Calls pre-post package hook
    """

    if conanfile.package_folder == conanfile.build_folder:
        raise ConanException("Cannot 'conan package' to the build folder. "
                             "--build-folder and package folder can't be the same")

    mkdir(conanfile.package_folder)
    scoped_output = conanfile.output
    # Make the copy of all the patterns
    scoped_output.info("Generating the package")
    scoped_output.info("Packaging in folder %s" % conanfile.package_folder)

    hook_manager.execute("pre_package", conanfile=conanfile)
    if hasattr(conanfile, "package"):
        scoped_output.highlight("Calling package()")
        with conanfile_exception_formatter(conanfile, "package"):
            with chdir(conanfile.build_folder):
                with conanfile_remove_attr(conanfile, ['info'], "package"):
                    conanfile.package()
    hook_manager.execute("post_package", conanfile=conanfile)

    save(os.path.join(conanfile.package_folder, CONANINFO), conanfile.info.dumps())
    manifest = FileTreeManifest.create(conanfile.package_folder)
    manifest.save(conanfile.package_folder)
    package_output = ConanOutput(scope="%s: package()" % scoped_output.scope)
    manifest.report_summary(package_output, "Packaged")

    prev = manifest.summary_hash
    scoped_output.info("Created package revision %s" % prev)
    pref = PkgReference(ref, package_id)
    pref.revision = prev
    scoped_output.success("Package '%s' created" % package_id)
    scoped_output.success("Full package reference: {}".format(pref.repr_notime()))
    return prev


def run_configure_method(conanfile, down_options, profile_options, ref):
    """ Run all the config-related functions for the given conanfile object """

    initial_requires_count = len(conanfile.requires)

    if hasattr(conanfile, "config_options"):
        with conanfile_exception_formatter(conanfile, "config_options"):
            conanfile.config_options()
    elif "auto_shared_fpic" in conanfile.implements:
        auto_shared_fpic_config_options(conanfile)

    auto_language(conanfile)  # default implementation removes `compiler.cstd`

    # Assign only the current package options values, but none of the dependencies
    is_consumer = conanfile._conan_is_consumer  # noqa
    conanfile.options.apply_downstream(down_options, profile_options, ref, is_consumer)

    if hasattr(conanfile, "configure"):
        with conanfile_exception_formatter(conanfile, "configure"):
            conanfile.configure()
    elif "auto_shared_fpic" in conanfile.implements:
        auto_shared_fpic_configure(conanfile)

    if initial_requires_count != len(conanfile.requires):
        conanfile.output.warning("Requirements should only be added in the requirements()/"
                                 "build_requirements() methods, not configure()/config_options(), "
                                 "which might raise errors in the future.", warn_tag="deprecated")

    result = conanfile.options.get_upstream_options(down_options, ref, is_consumer)
    self_options, up_options, private_up_options = result
    # self_options are the minimum to reproduce state, as defined from downstream (not profile)
    conanfile.self_options = self_options
    # up_options are the minimal options that should be propagated to dependencies
    conanfile.up_options = up_options
    conanfile.private_up_options = private_up_options

    PackageType.compute_package_type(conanfile)

    conanfile.build_requires = BuildRequirements(conanfile.requires)
    conanfile.test_requires = TestRequirements(conanfile.requires)
    conanfile.tool_requires = ToolRequirements(conanfile.requires)

    if hasattr(conanfile, "requirements"):
        with conanfile_exception_formatter(conanfile, "requirements"):
            conanfile.requirements()

    if hasattr(conanfile, "build_requirements"):
        with conanfile_exception_formatter(conanfile, "build_requirements"):
            conanfile.build_requirements()


def auto_shared_fpic_config_options(conanfile):
    if conanfile.settings.get_safe("os") == "Windows":
        conanfile.options.rm_safe("fPIC")


def auto_shared_fpic_configure(conanfile):
    if conanfile.options.get_safe("header_only"):
        conanfile.options.rm_safe("fPIC")
        conanfile.options.rm_safe("shared")
    elif conanfile.options.get_safe("shared"):
        conanfile.options.rm_safe("fPIC")


def auto_header_only_package_id(conanfile):
    if conanfile.options.get_safe("header_only") or conanfile.package_type is PackageType.HEADER:
        conanfile.info.clear()


def auto_language(conanfile):
    if not conanfile.languages:
        conanfile.settings.rm_safe("compiler.cstd")
        return
    if "C" not in conanfile.languages:
        conanfile.settings.rm_safe("compiler.cstd")
    if "C++" not in conanfile.languages:
        conanfile.settings.rm_safe("compiler.cppstd")
        conanfile.settings.rm_safe("compiler.libcxx")
