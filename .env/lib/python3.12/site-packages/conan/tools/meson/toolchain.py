import os
import textwrap

from jinja2 import Template, StrictUndefined

from conan.errors import ConanException
from conan.internal import check_duplicated_generator
from conan.internal.internal_tools import raise_on_universal_arch
from conan.tools.apple.apple import is_apple_os, apple_min_version_flag, \
    resolve_apple_flags, apple_extra_flags
from conan.tools.build.cross_building import cross_building, can_run
from conan.tools.build.flags import (architecture_link_flag, libcxx_flags, architecture_flag,
                                     threads_flags)
from conan.tools.env import VirtualBuildEnv
from conan.tools.meson.helpers import get_apple_subsystem, to_cppstd_flag, to_cstd_flag, \
    to_meson_machine, to_meson_value
from conan.tools.microsoft import VCVars, msvc_runtime_flag
from conan.internal.util.files import save


class MesonToolchain:
    """
    MesonToolchain generator
    """
    native_filename = "conan_meson_native.ini"
    cross_filename = "conan_meson_cross.ini"

    _meson_file_template = textwrap.dedent("""\
    [properties]
    {% for it, value in properties.items() -%}
    {{it}} = {{value}}
    {% endfor %}

    [constants]
    preprocessor_definitions = [{% for it, value in preprocessor_definitions.items() -%}
    '-D{{ it }}="{{ value}}"'{%- if not loop.last %}, {% endif %}{% endfor %}]

    [project options]
    {% for it, value in project_options.items() -%}
    {{it}} = {{value}}
    {% endfor %}

    {% for subproject, listkeypair in subproject_options -%}
    [{{subproject}}:project options]
    {% for keypair in listkeypair -%}
    {% for it, value in keypair.items() -%}
    {{it}} = {{value}}
    {% endfor %}
    {% endfor %}
    {% endfor %}

    [binaries]
    {% if c %}
    c = {{c}}
    {% endif %}
    {% if cpp %}
    cpp = {{cpp}}
    {% endif %}
    {% if ld %}
    ld = {{ld}}
    {% endif %}
    {% if is_apple_system %}
    {% if objc %}
    objc = '{{objc}}'
    {% endif %}
    {% if objcpp %}
    objcpp = '{{objcpp}}'
    {% endif %}
    {% endif %}
    {% if c_ld %}
    c_ld = '{{c_ld}}'
    {% endif %}
    {% if cpp_ld %}
    cpp_ld = '{{cpp_ld}}'
    {% endif %}
    {% if ar %}
    ar = '{{ar}}'
    {% endif %}
    {% if strip %}
    strip = '{{strip}}'
    {% endif %}
    {% if as %}
    as = '{{as}}'
    {% endif %}
    {% if windres %}
    windres = '{{windres}}'
    {% endif %}
    {% if pkgconfig %}
    pkgconfig = '{{pkgconfig}}'
    {% endif %}
    {% if pkgconfig %}
    pkg-config = '{{pkgconfig}}'
    {% endif %}

    [built-in options]
    {% if buildtype %}
    buildtype = '{{buildtype}}'
    {% endif %}
    {% if default_library %}
    default_library = '{{default_library}}'
    {% endif %}
    {% if b_vscrt %}
    b_vscrt = '{{b_vscrt}}'
    {% endif %}
    {% if b_ndebug %}
    b_ndebug = {{b_ndebug}}
    {% endif %}
    {% if b_staticpic %}
    b_staticpic = {{b_staticpic}}
    {% endif %}
    {% if cpp_std %}
    cpp_std = '{{cpp_std}}'
    {% endif %}
    {% if c_std %}
    c_std = '{{c_std}}'
    {% endif %}
    {% if backend %}
    backend = '{{backend}}'
    {% endif %}
    {% if pkg_config_path %}
    pkg_config_path = '{{pkg_config_path}}'
    {% endif %}
    {% if build_pkg_config_path %}
    build.pkg_config_path = '{{build_pkg_config_path}}'
    {% endif %}
    # C/C++ arguments
    c_args = {{c_args}} + preprocessor_definitions
    c_link_args = {{c_link_args}}
    cpp_args = {{cpp_args}} + preprocessor_definitions
    cpp_link_args = {{cpp_link_args}}
    {% if is_apple_system %}
    # Objective-C/C++ arguments
    objc_args = {{objc_args}} + preprocessor_definitions
    objc_link_args = {{objc_link_args}}
    objcpp_args = {{objcpp_args}} + preprocessor_definitions
    objcpp_link_args = {{objcpp_link_args}}
    {% endif %}

    {% for context, values in cross_build.items() %}
    [{{context}}_machine]
    system = '{{values["system"]}}'
    {% if values.get("subsystem") %}
    subsystem = '{{values["subsystem"]}}'
    {% endif %}
    cpu_family = '{{values["cpu_family"]}}'
    cpu = '{{values["cpu"]}}'
    endian = '{{values["endian"]}}'
    {% endfor %}
    """)

    def __init__(self, conanfile, backend=None, native=False):
        """
        :param conanfile: ``< ConanFile object >`` The current recipe object. Always use ``self``.
        :param backend: (**DEPRECATED**, use ``self.backend`` instead) ``str`` ``backend`` Meson variable
                        value. By default, ``ninja``.
        :param native: ``bool`` Indicates whether you want Conan to create the
                       ``conan_meson_native.ini`` in a cross-building context. Notice that it only
                       makes sense if your project's ``meson.build`` uses the ``native=true``
                       (see also https://mesonbuild.com/Cross-compilation.html#mixing-host-and-build-targets).
        """
        raise_on_universal_arch(conanfile)
        self._conanfile = conanfile
        self._native = native
        self._is_apple_system = is_apple_os(self._conanfile)
        is_cross_building = cross_building(conanfile)  # x86_64->x86 is considered cross-building
        if not is_cross_building and native:
            raise ConanException("You can only pass native=True if you're cross-building, "
                                 "otherwise, it could cause unexpected results.")
        self._conanfile_conf = self._conanfile.conf_build if native else self._conanfile.conf
        # Values are kept as Python built-ins so users can modify them more easily, and they are
        # only converted to Meson file syntax for rendering
        # priority: first user conf, then recipe, last one is default "ninja"
        #: Backend to use. Defined by the conf ``tools.meson.mesontoolchain:backend``. By default, ``ninja``.
        self.backend = backend or 'ninja'
        build_type = self._conanfile.settings.get_safe("build_type")
        #: Build type to use.
        self.buildtype = {"Debug": "debug",  # Note, it is not "'debug'"
                          "Release": "release",
                          "MinSizeRel": "minsize",
                          "RelWithDebInfo": "debugoptimized"}.get(build_type, build_type)
        #: Disable asserts.
        self.b_ndebug = "true" if self.buildtype != "debug" else "false"

        # https://mesonbuild.com/Builtin-options.html#base-options
        fpic = self._conanfile.options.get_safe("fPIC")
        shared = self._conanfile.options.get_safe("shared")
        #: Build static libraries as position independent. By default, ``self.options.get_safe("fPIC")``
        self.b_staticpic = fpic if (shared is False and fpic is not None) else None
        # https://mesonbuild.com/Builtin-options.html#core-options
        # Do not adjust "debug" if already adjusted "buildtype"
        #: Default library type, e.g., "shared.
        self.default_library = ("shared" if shared else "static") if shared is not None else None

        compiler = self._conanfile.settings.get_safe("compiler")
        if compiler is None:
            raise ConanException("MesonToolchain needs 'settings.compiler', but it is not defined")
        compiler_version = self._conanfile.settings.get_safe("compiler.version")
        if compiler_version is None:
            raise ConanException("MesonToolchain needs 'settings.compiler.version', but it is not defined")

        cppstd = self._conanfile.settings.get_safe("compiler.cppstd")
        cstd = self._conanfile.settings.get_safe("compiler.cstd")
        #: C++ language standard to use. Defined by ``to_cppstd_flag()`` by default.
        self.cpp_std = to_cppstd_flag(self._conanfile, compiler, compiler_version, cppstd)
        #: C language standard to use. Defined by ``to_cstd_flag()`` by default.
        self.c_std = to_cstd_flag(self._conanfile, cstd)
        #: VS runtime library to use. Defined by ``msvc_runtime_flag()`` by default.
        self.b_vscrt = None
        if compiler in ("msvc", "clang"):
            vscrt = msvc_runtime_flag(self._conanfile)
            if vscrt:
                self.b_vscrt = str(vscrt).lower()

        # Extra flags
        #: List of extra ``CXX`` flags. Added to ``cpp_args``
        self.extra_cxxflags = []
        #: List of extra ``C`` flags. Added to ``c_args``
        self.extra_cflags = []
        #: List of extra linker flags. Added to ``c_link_args`` and ``cpp_link_args``
        self.extra_ldflags = []
        #: List of extra preprocessor definitions. Added to ``c_args`` and ``cpp_args`` with the
        #: format ``-D[FLAG_N]``.
        self.extra_defines = []
        #: Architecture flag deduced by Conan and added to ``c_args``, ``cpp_args``, ``c_link_args`` and ``cpp_link_args``
        self.arch_flag = architecture_flag(self._conanfile)  # https://github.com/conan-io/conan/issues/17624
        #: Architecture link flag deduced by Conan and added to ``c_link_args`` and ``cpp_link_args``
        self.arch_link_flag = architecture_link_flag(self._conanfile)
        #: Threads flags deduced by Conan and added to ``c_args``, ``cpp_args``, ``c_link_args`` and ``cpp_link_args``
        self.threads_flags = threads_flags(self._conanfile)
        #: Dict-like object that defines Meson ``properties`` with ``key=value`` format
        self.properties = {}
        #: Dict-like object that defines Meson ``project options`` with ``key=value`` format
        self.project_options = {
            "wrap_mode": "nofallback"  # https://github.com/conan-io/conan/issues/10671
        }
        #: Dict-like object that defines Meson ``preprocessor definitions``
        self.preprocessor_definitions = {}
        # Add all the default dirs
        self.project_options.update(self._get_default_dirs())
        #: Dict-like object that defines Meson ``subproject options``.
        self.subproject_options = {}
        #: Defines the Meson ``pkg_config_path`` variable
        self.pkg_config_path = self._conanfile.generators_folder
        #: Defines the Meson ``build.pkg_config_path`` variable (build context)
        # Issue: https://github.com/conan-io/conan/issues/12342
        # Issue: https://github.com/conan-io/conan/issues/14935
        self.build_pkg_config_path = None
        self.libcxx, self.gcc_cxx11_abi = libcxx_flags(self._conanfile)
        #: Dict-like object with the build, host, and target as the Meson machine context
        self.cross_build = {}
        default_comp = ""
        default_comp_cpp = ""
        if native is False and is_cross_building:
            os_host = conanfile.settings.get_safe("os")
            arch_host = conanfile.settings.get_safe("arch")
            os_build = conanfile.settings_build.get_safe('os')
            arch_build = conanfile.settings_build.get_safe('arch')
            self.cross_build["build"] = to_meson_machine(os_build, arch_build)
            self.cross_build["host"] = to_meson_machine(os_host, arch_host)
            # Check subsystem if it's Apple cross-building only. It requires Meson >= 1.2.0,
            # but it does not break lower versions.
            # See https://mesonbuild.com/Reference-tables.html#subsystem-names-since-120
            # Issue: https://github.com/conan-io/conan/issues/17873
            if self._is_apple_system and is_apple_os(conanfile, build_context=True):
                sdk_build = conanfile.settings_build.get_safe("os.sdk")
                sdk_host = conanfile.settings.get_safe("os.sdk")
                self.cross_build["host"]["subsystem"] = get_apple_subsystem(sdk_host)
                self.cross_build["build"]["subsystem"] = get_apple_subsystem(sdk_build)
            # Issue: https://github.com/conan-io/conan/issues/19217
            self.properties["needs_exe_wrapper"] = not can_run(self._conanfile)
            if hasattr(conanfile, 'settings_target') and conanfile.settings_target:
                settings_target = conanfile.settings_target
                os_target = settings_target.get_safe("os")
                arch_target = settings_target.get_safe("arch")
                self.cross_build["target"] = to_meson_machine(os_target, arch_target)
            if is_apple_os(self._conanfile):  # default cross-compiler in Apple is common
                default_comp = "clang"
                default_comp_cpp = "clang++"
        else:
            if "clang" in compiler:
                default_comp = "clang"
                default_comp_cpp = "clang++"
            elif compiler == "gcc":
                default_comp = "gcc"
                default_comp_cpp = "g++"

        if "Visual" in compiler or compiler == "msvc":
            default_comp = "cl"
            default_comp_cpp = "cl"

        # Read configuration for sys_root property (honoring existing conf)
        self._sys_root = self._conanfile_conf.get("tools.build:sysroot", check_type=str)

        # Read configuration for compilers
        compilers_by_conf = self._conanfile_conf.get("tools.build:compiler_executables", default={},
                                                     check_type=dict)
        # Read the VirtualBuildEnv to update the variables
        build_env = self._conanfile.buildenv_build.vars(self._conanfile) if native else (
            VirtualBuildEnv(self._conanfile, auto_generate=True).vars())
        #: Sets the Meson ``c`` variable, defaulting to the ``CC`` build environment value.
        #: If provided as a blank-separated string, it will be transformed into a list.
        #: Otherwise, it remains a single string.
        self.c = compilers_by_conf.get("c") or self._sanitize_env_format(build_env.get("CC")) or default_comp
        #: Sets the Meson ``cpp`` variable, defaulting to the ``CXX`` build environment value.
        #: If provided as a blank-separated string, it will be transformed into a list.
        #: Otherwise, it remains a single string.
        self.cpp = compilers_by_conf.get("cpp") or self._sanitize_env_format(build_env.get("CXX")) or default_comp_cpp
        #: Sets the Meson ``ld`` variable, defaulting to the ``LD`` build environment value.
        #: If provided as a blank-separated string, it will be transformed into a list.
        #: Otherwise, it remains a single string.
        self.ld = self._sanitize_env_format(build_env.get("LD"))
        # FIXME: Should we use the new tools.build:compiler_executables and avoid buildenv?
        # Issue related: https://github.com/mesonbuild/meson/issues/6442
        # PR related: https://github.com/mesonbuild/meson/pull/6457
        #: Defines the Meson ``c_ld`` variable. Defaulted to ``CC_LD``
        #: environment value
        self.c_ld = build_env.get("CC_LD")
        #: Defines the Meson ``cpp_ld`` variable. Defaulted to ``CXX_LD``
        #: environment value
        self.cpp_ld = build_env.get("CXX_LD")
        #: Defines the Meson ``ar`` variable. Defaulted to ``AR`` build environment value
        self.ar = build_env.get("AR")
        #: Defines the Meson ``strip`` variable. Defaulted to ``STRIP`` build environment value
        self.strip = build_env.get("STRIP")
        #: Defines the Meson ``as`` variable. Defaulted to ``AS`` build environment value
        self.as_ = build_env.get("AS")
        #: Defines the Meson ``windres`` variable. Defaulted to ``WINDRES`` build environment value
        self.windres = build_env.get("WINDRES")
        #: Defines the Meson ``pkgconfig`` variable. Defaulted to ``PKG_CONFIG``
        #: build environment value
        self.pkgconfig = (self._conanfile_conf.get("tools.gnu:pkg_config", check_type=str) or
                          build_env.get("PKG_CONFIG"))
        #: Defines the Meson ``c_args`` variable. Defaulted to ``CFLAGS`` build environment value
        self.c_args = self._get_env_list(build_env.get("CFLAGS", []))
        #: Defines the Meson ``c_link_args`` variable. Defaulted to ``LDFLAGS`` build
        #: environment value
        self.c_link_args = self._get_env_list(build_env.get("LDFLAGS", []))
        #: Defines the Meson ``cpp_args`` variable. Defaulted to ``CXXFLAGS`` build environment value
        self.cpp_args = self._get_env_list(build_env.get("CXXFLAGS", []))
        #: Defines the Meson ``cpp_link_args`` variable. Defaulted to ``LDFLAGS`` build
        #: environment value
        self.cpp_link_args = self._get_env_list(build_env.get("LDFLAGS", []))

        # Apple flags and variables
        #: Apple arch flag as a list, e.g., ``["-arch", "i386"]``
        self.apple_arch_flag = []
        #: Apple sysroot flag as a list, e.g., ``["-isysroot", "./Platforms/MacOSX.platform"]``
        self.apple_isysroot_flag = []
        #: Apple minimum binary version flag as a list, e.g., ``["-mios-version-min", "10.8"]``
        self.apple_min_version_flag = []
        #: Apple bitcode, visibility and arc flags
        self.apple_extra_flags = apple_extra_flags(self._conanfile)
        #: Defines the Meson ``objc`` variable. Defaulted to ``None``, if if any Apple OS ``clang``
        self.objc = None
        #: Defines the Meson ``objcpp`` variable. Defaulted to ``None``, if if any Apple OS ``clang++``
        self.objcpp = None
        #: Defines the Meson ``objc_args`` variable. Defaulted to ``OBJCFLAGS`` build environment value
        self.objc_args = []
        #: Defines the Meson ``objc_link_args`` variable. Defaulted to ``LDFLAGS`` build environment value
        self.objc_link_args = []
        #: Defines the Meson ``objcpp_args`` variable. Defaulted to ``OBJCXXFLAGS`` build environment value
        self.objcpp_args = []
        #: Defines the Meson ``objcpp_link_args`` variable. Defaulted to ``LDFLAGS`` build environment value
        self.objcpp_link_args = []

        self._resolve_apple_flags_and_variables(build_env, compilers_by_conf)
        if native is False:
            self._resolve_android_cross_compilation()

    def _get_default_dirs(self):
        """
        Get all the default directories from cpp.package.

        Issues related:
            - https://github.com/conan-io/conan/issues/9713
            - https://github.com/conan-io/conan/issues/11596
        """
        def _get_cpp_info_value(name):
            elements = getattr(self._conanfile.cpp.package, name)
            return elements[0] if elements else None

        ret = {}
        bindir = _get_cpp_info_value("bindirs")
        datadir = _get_cpp_info_value("resdirs")
        libdir = _get_cpp_info_value("libdirs")
        includedir = _get_cpp_info_value("includedirs")
        if bindir:
            ret.update({
                'bindir': bindir,
                'sbindir': bindir,
                'libexecdir': bindir
            })
        if datadir:
            ret.update({
                'datadir': datadir,
                'localedir': datadir,
                'mandir': datadir,
                'infodir': datadir
            })
        if includedir:
            ret["includedir"] = includedir
        if libdir:
            ret["libdir"] = libdir
        return ret

    def _resolve_apple_flags_and_variables(self, build_env, compilers_by_conf):
        if not self._is_apple_system:
            return
        # Calculating the main Apple flags
        min_flag, arch_flag, isysroot_flag = (
            resolve_apple_flags(self._conanfile, is_cross_building=self.cross_build))
        self.apple_arch_flag = arch_flag.split() if arch_flag else []
        self.apple_isysroot_flag = isysroot_flag.split() if isysroot_flag else []
        self.apple_min_version_flag = [apple_min_version_flag(self._conanfile)]
        # Objective C/C++ ones
        flags = []
        self.objc = compilers_by_conf.get("objc", "clang")
        self.objcpp = compilers_by_conf.get("objcpp", "clang++")
        enable_arc = self._conanfile.conf.get("tools.apple:enable_arc", check_type=bool)
        fobj_arc = ""
        if enable_arc:
            fobj_arc = "-fobjc-arc"
        if enable_arc is False:
            fobj_arc = "-fno-objc-arc"
        self.objc_args = self._get_env_list(build_env.get('OBJCFLAGS', [])) + [fobj_arc]
        self.objc_link_args = self._get_env_list(build_env.get('LDFLAGS', []))
        self.objcpp_args = self._get_env_list(build_env.get('OBJCXXFLAGS', [])) + [fobj_arc]
        self.objcpp_link_args = self._get_env_list(build_env.get('LDFLAGS', []))

    def _resolve_android_cross_compilation(self):
        if not self.cross_build or not self.cross_build["host"]["system"] == "android":
            return

        ndk_path = self._conanfile_conf.get("tools.android:ndk_path")
        if not ndk_path:
            raise ConanException("You must provide a NDK path. Use 'tools.android:ndk_path' "
                                 "configuration field.")

        arch = self._conanfile.settings.get_safe("arch")
        os_build = self.cross_build["build"]["system"]
        ndk_bin = os.path.join(ndk_path, "toolchains",
                               "llvm", "prebuilt", "{}-x86_64".format(os_build), "bin")
        android_api_level = self._conanfile.settings.get_safe("os.api_level")
        android_target = {'armv7': 'armv7a-linux-androideabi',
                          'armv8': 'aarch64-linux-android',
                          'x86': 'i686-linux-android',
                          'x86_64': 'x86_64-linux-android'}.get(arch)
        os_build = self._conanfile.settings_build.get_safe('os')
        compile_ext = ".cmd" if os_build == "Windows" else ""
        # User has more prio than Conan
        self.c = os.path.join(ndk_bin, f"{android_target}{android_api_level}-clang{compile_ext}")
        self.cpp = os.path.join(ndk_bin, f"{android_target}{android_api_level}-clang++{compile_ext}")
        self.ar = os.path.join(ndk_bin, "llvm-ar")

    def _get_extra_flags(self):
        # Now, it's time to get all the flags defined by the user
        cxxflags = self._conanfile_conf.get("tools.build:cxxflags", default=[], check_type=list)
        cflags = self._conanfile_conf.get("tools.build:cflags", default=[], check_type=list)
        sharedlinkflags = self._conanfile_conf.get("tools.build:sharedlinkflags", default=[],
                                                   check_type=list)
        exelinkflags = self._conanfile_conf.get("tools.build:exelinkflags", default=[],
                                                check_type=list)
        linker_scripts = self._conanfile_conf.get("tools.build:linker_scripts", default=[],
                                                  check_type=list)
        linker_script_flags = ['-T' + linker_script for linker_script in linker_scripts]
        defines = self._conanfile_conf.get("tools.build:defines", default=[], check_type=list)
        sys_root = [f"--sysroot={self._sys_root}"] if self._sys_root else [""]
        ld = (sharedlinkflags + exelinkflags + linker_script_flags + sys_root + self.extra_ldflags
              + self.threads_flags)
        # Apple extra flags from confs (visibilty, bitcode, arc)
        cxxflags += self.apple_extra_flags
        cflags += self.apple_extra_flags
        ld += self.apple_extra_flags
        return {
            "cxxflags": [self.arch_flag] + cxxflags + sys_root + self.extra_cxxflags
                        + self.threads_flags,
            "cflags": [self.arch_flag] + cflags + sys_root + self.extra_cflags + self.threads_flags,
            "ldflags": [self.arch_flag] + [self.arch_link_flag] + ld,
            "defines": [f"-D{d}" for d in (defines + self.extra_defines)]
        }

    @staticmethod
    def _get_env_list(v):
        # FIXME: Should Environment have the "check_type=None" keyword as Conf?
        return v.strip().split() if not isinstance(v, list) else v

    @staticmethod
    def _filter_list_empty_fields(v):
        return list(filter(bool, v))

    @staticmethod
    def _sanitize_env_format(value):
        if value is None or isinstance(value, list):
            return value
        if not isinstance(value, str):
            raise ConanException(f"MesonToolchain: Value '{value}' should be a string")
        ret = [x.strip() for x in value.split() if x]
        return ret[0] if len(ret) == 1 else ret

    @property
    def _context(self):
        apple_flags = self.apple_isysroot_flag + self.apple_arch_flag + self.apple_min_version_flag
        extra_flags = self._get_extra_flags()

        self.c_args.extend(apple_flags + extra_flags["cflags"] + extra_flags["defines"])
        self.cpp_args.extend(apple_flags + extra_flags["cxxflags"] + extra_flags["defines"])
        self.c_link_args.extend(apple_flags + extra_flags["ldflags"])
        self.cpp_link_args.extend(apple_flags + extra_flags["ldflags"])
        # Objective C/C++
        self.objc_args.extend(self.c_args)
        self.objcpp_args.extend(self.cpp_args)
        # These link_args have already the LDFLAGS env value so let's add only the new possible ones
        self.objc_link_args.extend(self.c_link_args)
        self.objcpp_link_args.extend(self.cpp_link_args)

        if self.libcxx:
            self.cpp_args.append(self.libcxx)
            self.cpp_link_args.append(self.libcxx)
        if self.gcc_cxx11_abi:
            self.cpp_args.append("-D{}".format(self.gcc_cxx11_abi))

        subproject_options = {}
        for subproject, listkeypair in self.subproject_options.items():
            if listkeypair:
                if not isinstance(listkeypair, list):
                    raise ConanException("MesonToolchain.subproject_options must be a list of dicts")
                subproject_options[subproject] = [{k: to_meson_value(v) for k, v in keypair.items()}
                                                  for keypair in listkeypair]
        return {
            # https://mesonbuild.com/Machine-files.html#properties
            "properties": {k: to_meson_value(v) for k, v in self.properties.items()},
            # https://mesonbuild.com/Machine-files.html#project-specific-options
            "project_options": {k: to_meson_value(v) for k, v in self.project_options.items()},
            # https://mesonbuild.com/Subprojects.html#build-options-in-subproject
            "subproject_options": subproject_options.items(),
            # https://mesonbuild.com/Builtin-options.html#directories
            # https://mesonbuild.com/Machine-files.html#binaries
            # https://mesonbuild.com/Reference-tables.html#compiler-and-linker-selection-variables
            "c": to_meson_value(self.c),
            "cpp": to_meson_value(self.cpp),
            "ld": to_meson_value(self.ld),
            "objc": self.objc,
            "objcpp": self.objcpp,
            "c_ld": self.c_ld,
            "cpp_ld": self.cpp_ld,
            "ar": self.ar,
            "strip": self.strip,
            "as": self.as_,
            "windres": self.windres,
            "pkgconfig": self.pkgconfig,
            # https://mesonbuild.com/Builtin-options.html#core-options
            "buildtype": self.buildtype,
            "default_library": self.default_library,
            "backend": self._conanfile_conf.get("tools.meson.mesontoolchain:backend",
                                                default=self.backend),
            # https://mesonbuild.com/Builtin-options.html#base-options
            "b_vscrt": self.b_vscrt,
            "b_staticpic": to_meson_value(self.b_staticpic),  # boolean
            "b_ndebug": to_meson_value(self.b_ndebug),  # boolean as string
            # https://mesonbuild.com/Builtin-options.html#compiler-options
            "cpp_std": self.cpp_std,
            "c_std": self.c_std,
            "c_args": to_meson_value(self._filter_list_empty_fields(self.c_args)),
            "c_link_args": to_meson_value(self._filter_list_empty_fields(self.c_link_args)),
            "cpp_args": to_meson_value(self._filter_list_empty_fields(self.cpp_args)),
            "cpp_link_args": to_meson_value(self._filter_list_empty_fields(self.cpp_link_args)),
            "objc_args": to_meson_value(self._filter_list_empty_fields(self.objc_args)),
            "objc_link_args": to_meson_value(self._filter_list_empty_fields(self.objc_link_args)),
            "objcpp_args": to_meson_value(self._filter_list_empty_fields(self.objcpp_args)),
            "objcpp_link_args": to_meson_value(self._filter_list_empty_fields(self.objcpp_link_args)),
            "pkg_config_path": self.pkg_config_path,
            "build_pkg_config_path": self.build_pkg_config_path,
            "preprocessor_definitions": self.preprocessor_definitions,
            "cross_build": self.cross_build,
            "is_apple_system": self._is_apple_system
        }

    @property
    def _filename(self):
        if self.cross_build and self._native:
            return self.native_filename
        elif self.cross_build:
            return self.cross_filename
        else:
            return self.native_filename

    @property
    def _content(self):
        """
        Gets content of the file to be used by Meson as its context.

        :return: ``str`` whole Meson context content.
        """
        context = self._context
        content = Template(self._meson_file_template, trim_blocks=True, lstrip_blocks=True,
                           undefined=StrictUndefined).render(context)
        return content

    def generate(self):
        """
        Creates a ``conan_meson_native.ini`` (if native builds) or a
        ``conan_meson_cross.ini`` (if cross builds) with the proper content.
        If Windows OS, it will be created a ``conanvcvars.bat`` as well.
        """
        check_duplicated_generator(self, self._conanfile)
        self._conanfile.output.info(f"MesonToolchain generated: {self._filename}")
        save(self._filename, self._content)
        # FIXME: Should we check the OS and compiler to call VCVars?
        VCVars(self._conanfile).generate()
