import os
import shutil
import platform
import textwrap

from jinja2 import Template
from conan.internal import check_duplicated_generator
from conan.errors import ConanException
from conan.tools.env import VirtualBuildEnv
from conan.tools.microsoft import msvs_toolset
from conan.tools.microsoft.visual import vs_installation_path, _vcvars_path, _vcvars_versions
from conan.tools.qbs import common
from conan.internal.util.files import save


def _find_msvc(conanfile):
    vs_install_path = conanfile.conf.get("tools.microsoft.msbuild:installation_path")
    vs_version, vcvars_ver = _vcvars_versions(conanfile)
    vs_path = vs_install_path or vs_installation_path(vs_version)
    if vs_path is None or vs_version is None:
        return None

    vc_install_dir = os.path.join(vs_path, 'VC', 'Tools', 'MSVC')
    if not os.path.exists(vc_install_dir):
        return None
    compiler_versions = [v for v in os.listdir(vc_install_dir) if v.startswith(vcvars_ver)]
    compiler_versions.sort(reverse=True)

    build_arch_map = {
        'x86': 'Hostx86',
        'x86_64': 'Hostx64',
        'armv6': 'arm',
        'armv7': 'arm',
        'armv8': 'arm64',
    }
    host_arch_map = {
        'x86': 'x86',
        'x86_64': 'x64',
        'armv6': 'arm',
        'armv7': 'arm',
        'armv8': 'arm64',
    }
    build_arch = build_arch_map.get(str(conanfile.settings_build.arch))

    host_arch = host_arch_map.get(str(conanfile.settings.arch))

    if not host_arch or not build_arch:
        return None

    def cl_path(version):
        return os.path.join(vc_install_dir, version, 'bin', build_arch, host_arch, 'cl.exe')

    compiler_paths = [cl_path(version) for version in compiler_versions]
    compiler_paths = [p for p in compiler_paths if os.path.exists(p)]

    if len(compiler_paths) == 0:
        return None

    return compiler_paths[0]


def _find_clangcl(conanfile):
    vs_install_path = conanfile.conf.get("tools.microsoft.msbuild:installation_path")
    vs_version, _ = _vcvars_versions(conanfile)
    vs_path = vs_install_path or vs_installation_path(vs_version)

    compiler_path = os.path.join(vs_path, 'VC', 'Tools', 'Llvm', 'bin', 'clang-cl.exe')
    vcvars_path = _vcvars_path(vs_version, vs_install_path)
    if not os.path.exists(compiler_path):
        return None
    return compiler_path, vcvars_path


class _LinkerFlagsParser:
    def __init__(self, ld_flags):
        self.driver_linker_flags = []
        self.linker_flags = []

        for item in ld_flags:
            if item.startswith('-Wl'):
                self.linker_flags.extend(item.split(',')[1:])
            else:
                self.driver_linker_flags.append(item)


class QbsProfile:
    """
    Qbs profiles generator.

    This class generates file with the toolchain information that can be imported by Qbs.
    """

    def __init__(self, conanfile, profile='conan', default_profile='conan'):
        """
        :param conanfile: The current recipe object. Always use ``self``.
        :param profile: The name of the profile in settings. Defaults to ``"conan"``.
        :param default_profile: The name of the default profile. Defaults to ``"conan"``.

        """
        self._conanfile = conanfile
        self._profile = profile
        self._default_profile = default_profile

        self.extra_cflags = []
        self.extra_cxxflags = []
        self.extra_defines = []
        self.extra_sharedlinkflags = []
        self.extra_exelinkflags = []

        self._build_env = VirtualBuildEnv(self._conanfile, auto_generate=True).vars()

    @property
    def filename(self):
        """
        The name of the generated file. Returns ``qbs_settings.txt``.
        """
        return 'qbs_settings.txt'

    @property
    def content(self):
        """
        Returns the content of the settings file as dict of Qbs properties.
        """
        result = self._toolchain_properties()
        result.update(self._properties_from_settings())
        result.update(self._properties_from_conf())
        result.update(self._properties_from_options())
        # result.update(self._properties_from_env(self._build_env))
        return result

    def render(self):
        """
        Returns the content of the settings file as string.
        """
        template = textwrap.dedent('''\
            {%- for key, value in profile_values.items() %}
            profiles.{{profile}}.{{ key }}:{{ value }}
            {%- endfor %}
            defaultProfile: {{default_profile}}
        ''')
        t = Template(template)
        context = {
            'profile_values': self.content,
            'profile': self._profile,
            'default_profile': self._default_profile,
        }
        result = t.render(**context)
        return result

    def generate(self):
        """
        This method will save the generated files to the conanfile.generators_folder.

        Generates the "qbs_settings.txt" file. This file contains Qbs settings such as toolchain
        properties and can be imported using ``qbs config --import``.
        """
        check_duplicated_generator(self, self._conanfile)
        self._check_for_compiler()
        save(self.filename, self.render())

    def _check_for_compiler(self):
        compiler = self._conanfile.settings.get_safe('compiler')
        if not compiler:
            raise ConanException('Qbs: need compiler to be set in settings')

        if compiler not in ['msvc', 'gcc', 'clang', 'apple-clang']:
            raise ConanException(f'Qbs: compiler {compiler} not supported')

    def _get_qbs_toolchain(self):
        compiler = self._conanfile.settings.get_safe('compiler')
        the_os = self._conanfile.settings.get_safe('os')
        if the_os == 'Windows':
            if compiler == 'msvc':
                if msvs_toolset(self._conanfile) == 'ClangCL':
                    return 'clang-cl'
                return 'msvc'
            if compiler == 'gcc':
                return 'mingw'
            if compiler == 'clang':
                return 'clang-cl'
            raise ConanException('unknown windows compiler')
        if compiler == 'apple-clang':
            return 'xcode'
        # todo: other compilers?
        return compiler

    def _default_compiler_names(self, toolchain):
        if toolchain == 'msvc':
            return 'cl', 'cl'
        if toolchain == 'clang-cl':
            return 'clang-cl', 'clang-cl'
        if toolchain in ('gcc', 'mingw'):
            return 'gcc', 'g++'
        if toolchain in ('clang', 'xcode'):
            return 'clang', 'clang++'

        # what about other toolchains? IAR, Cosmic have a bunch of compilers based on arch
        return toolchain, toolchain

    def _find_exe(self, exe):
        if platform.system() == 'Windows':
            exe = exe + '.exe'
        if os.path.isabs(exe):
            return exe
        paths = self._build_env.get("PATH", "")
        for p in paths.split(os.pathsep):
            path = os.path.join(p, exe)
            if os.path.exists(path):
                return path
        return shutil.which(exe)

    def _toolchain_properties(self):
        toolchain = self._get_qbs_toolchain()
        the_os = self._conanfile.settings.get_safe('os')
        vcvars_path = None
        if the_os == 'Windows' and toolchain == 'msvc':
            compiler = _find_msvc(self._conanfile)
        elif the_os == 'Windows' and toolchain == 'clang-cl':
            compiler, vcvars_path = _find_clangcl(self._conanfile)
        else:
            # TODO: use CC also for msvc?
            c_compiler_default, cxx_compiler_default = self._default_compiler_names(toolchain)
            compilers_by_conf = self._conanfile.conf.get("tools.build:compiler_executables",
                                                         default={}, check_type=dict)
            c_compiler = (
                compilers_by_conf.get("c") or self._build_env.get("CC") or c_compiler_default)
            c_compiler = self._find_exe(c_compiler)

            cxx_compiler = (
                compilers_by_conf.get("cpp") or self._build_env.get("CXX") or cxx_compiler_default)
            cxx_compiler = self._find_exe(cxx_compiler)
            compiler = cxx_compiler or c_compiler
        if compiler is None:
            raise ConanException('cannot find compiler')

        result = {
            'qbs.toolchainType': toolchain,
            'cpp.compilerName': os.path.basename(compiler),
            'cpp.toolchainInstallPath': os.path.dirname(compiler).replace('\\', '/')
        }
        # GCC and friends also have separate props for compilers
        gcc_toolchains = ['clang', 'gcc', 'llvm', 'mingw', 'qcc', 'xcode']
        if toolchain in gcc_toolchains:
            result['cpp.cCompilerName'] = os.path.basename(c_compiler)
            result['cpp.cxxCompilerName'] = os.path.basename(cxx_compiler)
        if vcvars_path:
            result["cpp.vcvarsallPath"] = vcvars_path
        return result

    def _properties_from_settings(self):
        result = {}

        def map_qbs_property(key, qbs_property, value_map, fallback=None):
            value = value_map.get(self._conanfile.settings.get_safe(key)) or fallback
            if value is not None:
                result[qbs_property] = value

        map_qbs_property('arch', 'qbs.architecture', common.architecture_map)
        map_qbs_property('os', 'qbs.targetPlatform', common.target_platform_map, "undefined")
        map_qbs_property('build_type', 'qbs.buildVariant', common.build_variant_map)
        map_qbs_property(
            'compiler.cppstd', 'cpp.cxxLanguageVersion', common.cxx_language_version_map)
        map_qbs_property('compiler.runtime', 'cpp.runtimeLibrary', common.runtime_library_map)

        return result

    def _properties_from_options(self):
        result = {}

        def maybe_bool_str(b):
            return None if b is None else str(b).lower()

        fpic = maybe_bool_str(self._conanfile.options.get_safe('fPIC'))
        if fpic:
            result["cpp.positionIndependentCode"] = fpic

        return result

    def _properties_from_conf(self):
        result = {}

        def map_list_property(key, qbs_property, extra):
            value = self._conanfile.conf.get(key, default=[], check_type=list)
            value.extend(extra)
            if len(value) > 0:
                result[qbs_property] = value

        map_list_property("tools.build:cflags", "cpp.cFlags", self.extra_cflags)
        map_list_property("tools.build:cxxflags", "cpp.cxxFlags", self.extra_cxxflags)
        map_list_property("tools.build:defines", "cpp.defines", self.extra_defines)

        def ldflags():
            conf = self._conanfile.conf
            result = conf.get("tools.build:sharedlinkflags", default=[], check_type=list)
            result.extend(self.extra_sharedlinkflags)
            result.extend(conf.get("tools.build:exelinkflags", default=[], check_type=list))
            result.extend(self.extra_exelinkflags)
            linker_scripts = conf.get("tools.build:linker_scripts", default=[], check_type=list)
            result.extend(["-T'" + linker_script + "'" for linker_script in linker_scripts])
            return result

        ld_flags = ldflags()
        if len(ld_flags) > 0:
            parser = _LinkerFlagsParser(ld_flags)
            result['cpp.linkerFlags'] = parser.linker_flags
            result['cpp.driverLinkerFlags'] = parser.driver_linker_flags

        sysroot = self._conanfile.conf.get("tools.build:sysroot")
        if sysroot is not None:
            sysroot = sysroot.replace("\\", "/")
            result['qbs.sysroot'] = sysroot

        return result
