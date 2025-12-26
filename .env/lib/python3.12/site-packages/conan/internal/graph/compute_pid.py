from collections import OrderedDict

from conan.internal.errors import conanfile_remove_attr, conanfile_exception_formatter
from conan.errors import ConanException, ConanInvalidConfiguration
from conan.internal.methods import auto_header_only_package_id
from conan.internal.model.info import (ConanInfo, RequirementsInfo, RequirementInfo,
                                       PythonRequiresInfo)


def compute_package_id(node, modes, config_version, hook_manager):
    """
    Compute the binary package ID of this node
    """
    conanfile = node.conanfile
    unknown_mode, non_embed_mode, embed_mode, python_mode, build_mode = modes
    python_requires = getattr(conanfile, "python_requires", None)
    if python_requires:
        python_requires = python_requires.info_requires()

    data = OrderedDict()
    build_data = OrderedDict()
    for require, transitive in node.transitive_deps.items():
        dep_node = transitive.node
        require.deduce_package_id_mode(conanfile.package_type, dep_node,
                                       non_embed_mode, embed_mode, build_mode, unknown_mode)
        if require.package_id_mode is not None:
            req_info = RequirementInfo(dep_node.pref.ref, dep_node.pref.package_id,
                                       require.package_id_mode)
            if require.build:
                build_data[require] = req_info
            else:
                data[require] = req_info

    if conanfile.vendor:  # Make the package_id fully independent of dependencies versions
        data, build_data = OrderedDict(), OrderedDict()  # TODO, cleaner, now minimal diff

    reqs_info = RequirementsInfo(data)
    build_requires_info = RequirementsInfo(build_data)
    python_requires = PythonRequiresInfo(python_requires, python_mode)
    try:
        copied_options = conanfile.options.copy_conaninfo_options()
    except ConanException as e:
        raise ConanException(f"{conanfile}: {e}")

    conanfile.info = ConanInfo(settings=conanfile.settings.copy_conaninfo_settings(),
                               options=copied_options,
                               reqs_info=reqs_info,
                               build_requires_info=build_requires_info,
                               python_requires=python_requires,
                               conf=conanfile.conf.copy_conaninfo_conf(),
                               config_version=config_version.copy() if config_version else None)
    conanfile.original_info = conanfile.info.clone()

    run_validate_package_id(conanfile, hook_manager)

    if conanfile.info.settings_target:
        # settings_target has beed added to conan package via package_id api
        conanfile.original_info.settings_target = conanfile.info.settings_target

    info = conanfile.info
    node.package_id = info.package_id()


def run_validate_package_id(conanfile, hook_manager):
    # IMPORTANT: This validation code must run before calling info.package_id(), to mark "invalid"
    if hasattr(conanfile, "validate_build"):
        with conanfile_exception_formatter(conanfile, "validate_build"):
            with conanfile_remove_attr(conanfile, ['cpp_info'], "validate_build"):
                try:
                    conanfile.validate_build()
                except ConanInvalidConfiguration as e:
                    # This 'cant_build' will be ignored if we don't have to build the node.
                    conanfile.info.cant_build = str(e)

    if hasattr(conanfile, "validate") or hook_manager.validate_hook:
        with conanfile_exception_formatter(conanfile, "validate"):
            with conanfile_remove_attr(conanfile, ['cpp_info'], "validate"):
                try:
                    if hook_manager.validate_hook:
                        hook_manager.execute("pre_validate", conanfile=conanfile)
                    if hasattr(conanfile, "validate"):
                        conanfile.validate()
                    if hook_manager.validate_hook:
                        hook_manager.execute("post_validate", conanfile=conanfile)
                except ConanInvalidConfiguration as e:
                    conanfile.info.invalid = str(e)

    # Once we are done, call package_id() to narrow and change possible values
    if hasattr(conanfile, "package_id"):
        with conanfile_exception_formatter(conanfile, "package_id"):
            with conanfile_remove_attr(conanfile, ['cpp_info', 'settings', 'options'], "package_id"):
                conanfile.package_id()
    elif "auto_header_only" in conanfile.implements:
        auto_header_only_package_id(conanfile)
    if hook_manager.post_package_id_hook:
        with conanfile_exception_formatter(conanfile, "package_id"):
            with conanfile_remove_attr(conanfile, ['cpp_info', 'settings', 'options'], "package_id"):
                hook_manager.execute("post_package_id", conanfile=conanfile)

    conanfile.info.validate()
