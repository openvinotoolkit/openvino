from conan.internal.graph.graph import CONTEXT_BUILD
from conan.errors import ConanException
from conan.internal.model.recipe_ref import ref_matches


def initialize_conanfile_profile(conanfile, profile_build, profile_host, base_context,
                                 is_build_require, ref=None, parent=None):
    """ this function fills conanfile information with the profile informaiton
    It is called for:
        - computing the root_node
           - GraphManager.load_consumer_conanfile, for "conan source" command
           - GraphManager._load_root_consumer for "conan install <path to conanfile>
           - GraphManager._load_root_test_package for "conan create ." with test_package folder
        - computing each graph node:
            GraphBuilder->create_new_node
    """
    # NOTE: Need the context, as conanfile.context NOT defined yet

    # settings_build=profile_build ALWAYS
    # host -(r)-> host => settings_host=profile_host, settings_target=None
    # host -(br)-> build => settings_host=profile_build, settings_target=profile_host
    # build(gcc) -(r)-> build(openssl/zlib) => settings_host=profile_build, settings_target=None
    # build(gcc) -(br)-> build(gcc) => settings_host=profile_build, settings_target=profile_build
    # profile host
    settings_host = _per_package_settings(conanfile, profile_host, ref)
    settings_build = _per_package_settings(conanfile, profile_build, ref)
    if is_build_require or base_context == CONTEXT_BUILD:
        _initialize_conanfile(conanfile, profile_build, settings_build.copy(), ref)
        conanfile.buildenv_build = None
        conanfile.conf_build = None
    else:
        _initialize_conanfile(conanfile, profile_host, settings_host, ref)
        # Host profile with some build profile information
        is_consumer = conanfile._conan_is_consumer  # noqa
        conanfile.buildenv_build = profile_build.buildenv.get_profile_env(ref, is_consumer)
        conanfile.conf_build = profile_build.conf.get_conanfile_conf(ref, is_consumer)
    conanfile.settings_build = settings_build
    conanfile.settings_target = None

    if is_build_require:
        if base_context == CONTEXT_BUILD:
            conanfile.settings_target = settings_build.copy()
        else:
            conanfile.settings_target = settings_host.copy()
    else:
        if base_context == CONTEXT_BUILD:
            # if parent is first level tool-requires, required by HOST context
            if parent is None or parent.settings_target is None:
                conanfile.settings_target = settings_host.copy()
            else:
                conanfile.settings_target = parent.settings_target.copy()


def _per_package_settings(conanfile, profile, ref):
    # Prepare the settings for the loaded conanfile
    # Mixing the global settings with the specified for that name if exist
    tmp_settings = profile.processed_settings.copy()
    package_settings_values = profile.package_settings_values

    if package_settings_values:
        pkg_settings = []

        for pattern, settings in package_settings_values.items():
            if ref_matches(ref, pattern, conanfile._conan_is_consumer):  # noqa
                pkg_settings.extend(settings)

        if pkg_settings:
            tmp_settings.update_values(pkg_settings)
            # if the global settings are composed with per-package settings, need to preprocess

    return tmp_settings


def _initialize_conanfile(conanfile, profile, settings, ref):
    try:
        settings.constrained(conanfile.settings)
    except Exception as e:
        raise ConanException("The recipe %s is constraining settings. %s" % (
            conanfile.display_name, str(e)))
    conanfile.settings = settings
    conanfile.settings._frozen = True
    conanfile._conan_buildenv = profile.buildenv
    conanfile._conan_runenv = profile.runenv
    # Maybe this can be done lazy too
    conanfile.conf = profile.conf.get_conanfile_conf(ref, conanfile._conan_is_consumer)  # noqa


def consumer_definer(conanfile, profile_host, profile_build):
    """ conanfile.txt does not declare settings, but it assumes it uses all the profile settings
    These settings are very necessary for helpers like generators to produce the right output
    """
    tmp_settings = profile_host.processed_settings.copy()
    package_settings_values = profile_host.package_settings_values

    for pattern, settings in package_settings_values.items():
        if ref_matches(ref=None, pattern=pattern, is_consumer=True):
            tmp_settings.update_values(settings)

    tmp_settings_build = profile_build.processed_settings.copy()
    package_settings_values_build = profile_build.package_settings_values

    for pattern, settings in package_settings_values_build.items():
        if ref_matches(ref=None, pattern=pattern, is_consumer=True):
            tmp_settings_build.update_values(settings)

    conanfile.settings = tmp_settings
    conanfile.settings_build = tmp_settings_build
    conanfile.settings_target = None
    conanfile.settings._frozen = True
    conanfile._conan_buildenv = profile_host.buildenv
    conanfile._conan_runenv = profile_host.runenv
    conanfile.conf = profile_host.conf.get_conanfile_conf(None, is_consumer=True)
    conanfile.conf_build = profile_build.conf.get_conanfile_conf(None, is_consumer=True)
