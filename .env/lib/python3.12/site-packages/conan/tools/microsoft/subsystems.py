from conan.internal.subsystems import deduce_subsystem, subsystem_path


def unix_path(conanfile, path, scope="build"):
    subsystem = deduce_subsystem(conanfile, scope=scope)
    return subsystem_path(subsystem, path)


def unix_path_package_info_legacy(conanfile, path, path_flavor=None):
    message = "The use of 'unix_path_legacy_compat' is deprecated in Conan 2.0 and does not " \
              "perform path conversions. This is retained for compatibility with Conan 1.x " \
              "and will be removed in a future version."
    conanfile.output.warning(message, warn_tag="deprecated")
    return path
