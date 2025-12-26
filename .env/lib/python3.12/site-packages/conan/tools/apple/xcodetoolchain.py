import textwrap

from conan.internal import check_duplicated_generator
from conan.tools.apple.apple import to_apple_arch, xcodebuild_deployment_target_key
from conan.tools.apple.xcodedeps import GLOBAL_XCCONFIG_FILENAME, GLOBAL_XCCONFIG_TEMPLATE, \
    _add_includes_to_file_or_create, _xcconfig_settings_filename, _xcconfig_conditional
from conan.internal.util.files import save


class XcodeToolchain(object):
    filename = "conantoolchain"
    extension = ".xcconfig"

    _vars_xconfig = textwrap.dedent("""\
        // Definition of toolchain variables
        {apple_deployment_target}
        {clang_cxx_library}
        {clang_cxx_language_standard}
        """)

    _flags_xconfig = textwrap.dedent("""\
        // Global flags
        {defines}
        {cflags}
        {cppflags}
        {ldflags}
        """)

    _agreggated_xconfig = textwrap.dedent("""\
        // Conan XcodeToolchain generated file
        // Includes all installed configurations

        """)

    def __init__(self, conanfile):
        self._conanfile = conanfile
        arch = conanfile.settings.get_safe("arch")
        self.architecture = to_apple_arch(self._conanfile, default=arch)
        self.configuration = conanfile.settings.build_type
        self.libcxx = conanfile.settings.get_safe("compiler.libcxx")
        self.os_version = conanfile.settings.get_safe("os.version")
        self._global_defines = self._conanfile.conf.get("tools.build:defines", default=[], check_type=list)
        self._global_cxxflags = self._conanfile.conf.get("tools.build:cxxflags", default=[], check_type=list)
        self._global_cflags = self._conanfile.conf.get("tools.build:cflags", default=[], check_type=list)
        sharedlinkflags = self._conanfile.conf.get("tools.build:sharedlinkflags", default=[], check_type=list)
        exelinkflags = self._conanfile.conf.get("tools.build:exelinkflags", default=[], check_type=list)
        self._global_ldflags = sharedlinkflags + exelinkflags

    def generate(self):
        check_duplicated_generator(self, self._conanfile)
        save(self._agreggated_xconfig_filename, self._agreggated_xconfig_content)
        save(self._vars_xconfig_filename, self._vars_xconfig_content)
        if self._check_if_extra_flags:
            save(self._flags_xcconfig_filename, self._flags_xcconfig_content)
        save(GLOBAL_XCCONFIG_FILENAME, self._global_xconfig_content)

    @property
    def _cppstd(self):
        from conan.tools.build.flags import cppstd_flag
        cppstd = cppstd_flag(self._conanfile)
        if cppstd.startswith("-std="):
            return cppstd[5:]
        return cppstd

    @property
    def _apple_deployment_target(self):
        deployment_target_key = xcodebuild_deployment_target_key(self._conanfile.settings.get_safe("os"))
        return '{}{}={}'.format(deployment_target_key,
                                _xcconfig_conditional(self._conanfile.settings, self.configuration),
                                self.os_version) if deployment_target_key and self.os_version else ""

    @property
    def _clang_cxx_library(self):
        return 'CLANG_CXX_LIBRARY{}={}'.format(_xcconfig_conditional(self._conanfile.settings,
                                                                     self.configuration),
                                               self.libcxx) if self.libcxx else ""

    @property
    def _clang_cxx_language_standard(self):
        return 'CLANG_CXX_LANGUAGE_STANDARD{}={}'.format(_xcconfig_conditional(self._conanfile.settings, self.configuration),
                                                         self._cppstd) if self._cppstd else ""

    @property
    def _vars_xconfig_filename(self):
        return "conantoolchain{}{}".format(_xcconfig_settings_filename(self._conanfile.settings,
                                                                       self.configuration),
                                           self.extension)

    @property
    def _vars_xconfig_content(self):
        ret = self._vars_xconfig.format(apple_deployment_target=self._apple_deployment_target,
                                        clang_cxx_library=self._clang_cxx_library,
                                        clang_cxx_language_standard=self._clang_cxx_language_standard)
        return ret

    @property
    def _agreggated_xconfig_content(self):
        return _add_includes_to_file_or_create(self._agreggated_xconfig_filename,
                                               self._agreggated_xconfig,
                                               [self._vars_xconfig_filename])

    @property
    def _global_xconfig_content(self):
        files_to_include = [self._agreggated_xconfig_filename]
        if self._check_if_extra_flags:
            files_to_include.append(self._flags_xcconfig_filename)
        content = _add_includes_to_file_or_create(GLOBAL_XCCONFIG_FILENAME, GLOBAL_XCCONFIG_TEMPLATE,
                                                  files_to_include)
        return content

    @property
    def _agreggated_xconfig_filename(self):
        return self.filename + self.extension

    @property
    def _check_if_extra_flags(self):
        return self._global_cflags or self._global_cxxflags or self._global_ldflags or self._global_defines

    @property
    def _flags_xcconfig_content(self):
        defines = "GCC_PREPROCESSOR_DEFINITIONS = $(inherited) {}".format(" ".join(self._global_defines)) if self._global_defines else ""
        cflags = "OTHER_CFLAGS = $(inherited) {}".format(" ".join(self._global_cflags)) if self._global_cflags else ""
        cppflags = "OTHER_CPLUSPLUSFLAGS = $(inherited) {}".format(" ".join(self._global_cxxflags)) if self._global_cxxflags else ""
        ldflags = "OTHER_LDFLAGS = $(inherited) {}".format(" ".join(self._global_ldflags)) if self._global_ldflags else ""
        ret = self._flags_xconfig.format(defines=defines, cflags=cflags, cppflags=cppflags, ldflags=ldflags)
        return ret

    @property
    def _flags_xcconfig_filename(self):
        return "conan_global_flags" + self.extension
