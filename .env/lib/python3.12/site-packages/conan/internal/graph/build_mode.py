from conan.errors import ConanException
from conan.internal.model.recipe_ref import ref_matches


class BuildMode:
    """ build_mode => ["*"] if user wrote "--build=*"
                   => ["hello", "bye"] if user wrote "--build hello --build bye"
                   => ["hello/0.1@foo/bar"] if user wrote "--build hello/0.1@foo/bar"
                   => ["!foo"] or ["~foo"] means exclude when building all from sources
    """
    def __init__(self, params):
        self.missing = False
        self.never = False
        self.cascade = False
        self.editable = False
        self.patterns = []
        self.build_missing_patterns = []
        self._build_missing_excluded = []
        self._build_compatible_patterns = []
        self._build_compatible_excluded = []
        self._excluded_patterns = []
        if params is None:
            return

        assert isinstance(params, list)
        assert len(params) > 0  # Not empty list

        for param in params:
            if param == "missing":
                self.missing = True
            elif param == "editable":
                self.editable = True
            elif param == "never":
                self.never = True
            elif param == "cascade":
                self.cascade = True
            else:
                if param.startswith("missing:"):
                    clean_pattern = param[len("missing:"):]
                    if clean_pattern and clean_pattern[0] in ["!", "~"]:
                        self._build_missing_excluded.append(clean_pattern[1:])
                    else:
                        self.build_missing_patterns.append(clean_pattern)
                elif param == "compatible":
                    self._build_compatible_patterns = ["*"]
                elif param.startswith("compatible:"):
                    clean_pattern = param[len("compatible:"):]
                    if clean_pattern and clean_pattern[0] in ["!", "~"]:
                        self._build_compatible_excluded.append(clean_pattern[1:])
                    else:
                        self._build_compatible_patterns.append(clean_pattern)
                else:
                    clean_pattern = param
                    if clean_pattern and clean_pattern[0] in ["!", "~"]:
                        self._excluded_patterns.append(clean_pattern[1:])
                    else:
                        self.patterns.append(clean_pattern)

            if self.never and (self.missing or self.patterns or self.cascade):
                raise ConanException("--build=never not compatible with other options")

    def forced(self, conan_file, ref, with_deps_to_build=False):
        # TODO: ref can be obtained from conan_file

        for pattern in self._excluded_patterns:
            if ref_matches(ref, pattern, is_consumer=conan_file._conan_is_consumer):  # noqa
                conan_file.output.info("Excluded build from source")
                return False

        if conan_file.build_policy == "never":  # this package has been export-pkg
            return False

        if self.never:
            return False

        if conan_file.build_policy == "always":
            raise ConanException("{}: build_policy='always' has been removed. "
                                 "Please use 'missing' only".format(conan_file))

        if self.cascade and with_deps_to_build:
            return True

        # Patterns to match, if package matches pattern, build is forced
        for pattern in self.patterns:
            if ref_matches(ref, pattern, is_consumer=conan_file._conan_is_consumer):  # noqa
                return True
        return False

    def allowed(self, conan_file):
        if self.never or conan_file.build_policy == "never":  # this package has been export-pkg
            return False
        if self.missing:
            return True
        if conan_file.build_policy == "missing":
            conan_file.output.info("Building package from source as defined by "
                                   "build_policy='missing'")
            return True
        if self.should_build_missing(conan_file):
            return True
        if self.allowed_compatible(conan_file):
            return True
        return False

    def allowed_compatible(self, conanfile):
        if self._build_compatible_excluded:
            for pattern in self._build_compatible_excluded:
                if ref_matches(conanfile.ref, pattern, is_consumer=False):
                    return False
            return True  # If it has not been excluded by the negated patterns, it is included

        for pattern in self._build_compatible_patterns:
            if ref_matches(conanfile.ref, pattern, is_consumer=conanfile._conan_is_consumer):  # noqa
                return True

    def should_build_missing(self, conanfile):
        if self._build_missing_excluded:
            for pattern in self._build_missing_excluded:
                if ref_matches(conanfile.ref, pattern, is_consumer=False):
                    return False
            return True  # If it has not been excluded by the negated patterns, it is included

        for pattern in self.build_missing_patterns:
            if ref_matches(conanfile.ref, pattern, is_consumer=conanfile._conan_is_consumer):  # noqa
                return True
