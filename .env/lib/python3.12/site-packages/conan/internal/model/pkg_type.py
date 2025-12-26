from enum import Enum

from conan.errors import ConanException


class PackageType(Enum):
    LIBRARY = "library"  # abstract type, should contain shared option to define
    STATIC = "static-library"
    SHARED = "shared-library"
    HEADER = "header-library"
    BUILD_SCRIPTS = "build-scripts"
    APP = "application"
    PYTHON = "python-require"
    CONF = "configuration"
    UNKNOWN = "unknown"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        # This is useful for comparing with string type at user code, like ``package_type == "xxx"``
        return super().__eq__(PackageType(other))

    @staticmethod
    def compute_package_type(conanfile):
        # This doesnt implement the header_only option without shared one. Users should define
        # their package_type as they wish in the configure() method

        def deduce_from_options():
            try:
                header = conanfile.options.header_only
            except ConanException:
                pass
            else:
                if header:
                    return PackageType.HEADER

            try:
                shared = conanfile.options.shared
            except ConanException:
                pass
            else:
                if shared:
                    return PackageType.SHARED
                else:
                    return PackageType.STATIC
            return PackageType.UNKNOWN

        conanfile_type = conanfile.package_type
        if conanfile_type is not None:  # Explicit definition in recipe
            try:
                conanfile_type = PackageType(conanfile_type)
            except ValueError:
                raise ConanException(f"{conanfile}: Invalid package type '{conanfile_type}'. "
                                     f"Valid types: {[i.value for i in PackageType]}")
            if conanfile_type is PackageType.LIBRARY:
                conanfile_type = deduce_from_options()
                if conanfile_type is PackageType.UNKNOWN:
                    raise ConanException(f"{conanfile}: Package type is 'library',"
                                         " but no 'shared' option declared")
            elif any(option in conanfile.options for option in ["shared", "header_only"]):
                conanfile.output.warning(f"{conanfile}: package_type '{conanfile_type}' is defined, "
                                         "but 'shared' and/or 'header_only' options are present. "
                                         "The package_type will have precedence over the options "
                                         "regardless of their value.")
            conanfile.package_type = conanfile_type
        else:  # automatic default detection with option shared/header-only
            conanfile.package_type = deduce_from_options()
