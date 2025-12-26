import copy
import hashlib
import numbers
import platform
import re
import os
import fnmatch
import textwrap

from collections import OrderedDict

from jinja2 import Environment, FileSystemLoader

from conan.errors import ConanException
from conan.internal.api.detect import detect_api
from conan.internal.cache.home_paths import HomePaths
from conan.internal.model.options import _PackageOption
from conan.internal.model.recipe_ref import ref_matches
from conan.internal.model.settings import SettingsItem
from conan.internal.util.files import load, save

BUILT_IN_CONFS = {
    "core:required_conan_version": "Raise if current version does not match the defined range.",
    "core:non_interactive": "Disable interactive user input, raises error if input necessary",
    "core:warnings_as_errors": "Treat warnings matching any of the patterns in this list as errors and then raise an exception. "
                               "Current warning tags are 'network', 'deprecated'",
    "core:skip_warnings": "Do not show warnings matching any of the patterns in this list. "
                          "Current warning tags are 'network', 'deprecated', 'experimental'",
    "core:default_profile": "Defines the default host profile ('default' by default)",
    "core:default_build_profile": "Defines the default build profile ('default' by default)",
    "core:allow_uppercase_pkg_names": "Temporarily (will be removed in 2.X) allow uppercase names",
    "core.version_ranges:resolve_prereleases": "Whether version ranges can resolve to pre-releases or not",
    "core.upload:retry": "Number of retries in case of failure when uploading to Conan server",
    "core.upload:retry_wait": "Seconds to wait between upload attempts to Conan server",
    "core.upload:parallel": "Number of concurrent threads to upload packages",
    "core.download:parallel": "Number of concurrent threads to download packages",
    "core.download:retry": "Number of retries in case of failure when downloading from Conan server",
    "core.download:retry_wait": "Seconds to wait between download attempts from Conan server",
    "core.download:download_cache": "Define path to a file download cache",
    "core.cache:storage_path": "Absolute path where the packages and database are stored",
    "core:update_policy": "(Legacy). If equal 'legacy' when multiple remotes, update based on order of remotes, only the timestamp of the first occurrence of each revision counts.",
    # Sources backup
    "core.sources:download_cache": "Folder to store the sources backup",
    "core.sources:download_urls": "List of URLs to download backup sources from",
    "core.sources:upload_url": "Remote URL to upload backup sources to",
    "core.sources:exclude_urls": "URLs which will not be backed up",
    "core.sources.patch:extra_path": "Extra path to search for patch files for conan create",
    # Package ID
    "core.package_id:default_unknown_mode": "By default, 'semver_mode'",
    "core.package_id:default_non_embed_mode": "By default, 'minor_mode'",
    "core.package_id:default_embed_mode": "By default, 'full_mode'",
    "core.package_id:default_python_mode": "By default, 'minor_mode'",
    "core.package_id:default_build_mode": "By default, 'None'",
    "core.package_id:config_mode": "How the 'config_version' affects binaries. By default 'None'",
    # General HTTP(python-requests) configuration
    "core.net.http:max_retries": "Maximum number of connection retries (requests library)",
    "core.net.http:timeout": "Number of seconds without response to timeout (requests library)",
    "core.net.http:no_proxy_match": "List of urls to skip from proxies configuration",
    "core.net.http:proxies": "Dictionary containing the proxy configuration",
    "core.net.http:cacert_path": "Path containing a custom Cacert file",
    "core.net.http:client_cert": "Path or tuple of files containing a client cert (and key)",
    "core.net.http:clean_system_proxy": "If defined, the proxies system env-vars will be discarded",
    # Gzip compression
    "core.gzip:compresslevel": "The Gzip compression level for Conan artifacts (default=9)",
    # Excluded from revision_mode = "scm" dirty and Git().is_dirty() checks
    "core.scm:excluded": "List of excluded patterns for builtin git dirty checks",
    "core.scm:local_url": "By default allows to store local folders as remote url, but not upload them. Use 'allow' for allowing upload and 'block' to completely forbid it",
    # Compatibility opt-in, to be removed in future versions as optimized behavior becomes default
    "core.graph:compatibility_mode": "(Experimental) Set this to 'optimized' to enable the improved compatibility behaviour when querying multiple compatible binaries in remotes",
    # Tools
    "tools.android:ndk_path": "Argument for the CMAKE_ANDROID_NDK",
    "tools.android:cmake_legacy_toolchain": "Define to explicitly pass ANDROID_USE_LEGACY_TOOLCHAIN_FILE in CMake toolchain",
    "tools.build:skip_test": "Do not execute CMake.test() and Meson.test() when enabled",
    "tools.build:download_source": "Force download of sources for every package",
    "tools.build:jobs": "Default compile jobs number -jX Ninja, Make, /MP VS (default: max CPUs)",
    "tools.build:sysroot": "Pass the --sysroot=<tools.build:sysroot> flag if available. (None by default)",
    "tools.build.cross_building:can_run": "(boolean) Indicates whether is possible to run a non-native app on the same architecture. It's used by 'can_run' tool",
    "tools.build.cross_building:cross_build": "(boolean) Decides whether cross-building or not regardless of arch/OS settings. Used by 'cross_building' tool",
    "tools.build:verbosity": "Verbosity of build systems if set. Possible values are 'quiet' and 'verbose'",
    "tools.compilation:verbosity": "Verbosity of compilation tools if set. Possible values are 'quiet' and 'verbose'",
    "tools.cmake.cmaketoolchain:generator": "User defined CMake generator to use instead of default",
    "tools.cmake.cmaketoolchain:find_package_prefer_config": "Argument for the CMAKE_FIND_PACKAGE_PREFER_CONFIG",
    "tools.cmake.cmaketoolchain:toolchain_file": "Use other existing file rather than conan_toolchain.cmake one",
    "tools.cmake.cmaketoolchain:user_toolchain": "Inject existing user toolchains at the beginning of conan_toolchain.cmake",
    "tools.cmake.cmaketoolchain:system_name": "Define CMAKE_SYSTEM_NAME in CMakeToolchain",
    "tools.cmake.cmaketoolchain:system_version": "Define CMAKE_SYSTEM_VERSION in CMakeToolchain",
    "tools.cmake.cmaketoolchain:system_processor": "Define CMAKE_SYSTEM_PROCESSOR in CMakeToolchain",
    "tools.cmake.cmaketoolchain:toolset_arch": "Toolset architecture to be used as part of CMAKE_GENERATOR_TOOLSET in CMakeToolchain",
    "tools.cmake.cmaketoolchain:toolset_cuda": "(Experimental) Path to a CUDA toolset to use, or version if installed at the system level",
    "tools.cmake.cmaketoolchain:presets_environment": "String to define wether to add or not the environment section to the CMake presets. Empty by default, will generate the environment section in CMakePresets. Can take values: 'disabled'.",
    "tools.cmake.cmaketoolchain:extra_variables": "Dictionary with variables to be injected in CMakeToolchain (potential override of CMakeToolchain defined variables)",
    "tools.cmake.cmaketoolchain:enabled_blocks": "Select the specific blocks to use in the conan_toolchain.cmake",
    "tools.cmake.cmaketoolchain:user_presets": "(Experimental) Select a different name instead of CMakeUserPresets.json, empty to disable",
    "tools.cmake.cmake_layout:build_folder_vars": "Settings and Options that will produce a different build folder and different CMake presets names",
    "tools.cmake.cmake_layout:build_folder": "(Experimental) Allow configuring the base folder of the build for local builds",
    "tools.cmake.cmake_layout:test_folder": "(Experimental) Allow configuring the base folder of the build for test_package",
    "tools.cmake:cmake_program": "Path to CMake executable",
    "tools.cmake.cmakedeps:new": "Use the new CMakeDeps generator",
    "tools.cmake:ctest_args": "To inject list of arguments to CMake.ctest() runner",
    "tools.cmake:install_strip": "(Deprecated) Add --strip to cmake.install(). Use tools.build:install_strip instead",
    "tools.deployer:symlinks": "Set to False to disable deployers copying symlinks",
    "tools.files.download:retry": "Number of retries in case of failure when downloading",
    "tools.files.download:retry_wait": "Seconds to wait between download attempts",
    "tools.files.download:verify": "If set, overrides recipes on whether to perform SSL verification for their downloaded files. Only recommended to be set while testing",
    "tools.files.unzip:filter": "Define tar extraction filter: 'fully_trusted', 'tar', 'data'",
    "tools.graph:vendor": "(Experimental) If 'build', enables the computation of dependencies of vendoring packages to build them",
    "tools.graph:skip_binaries": "Allow the graph to skip binaries not needed in the current configuration (True by default)",
    "tools.graph:skip_build": "(Experimental) Do not expand build/tool_requires",
    "tools.graph:skip_test": "(Experimental) Do not expand test_requires. If building it might need 'tools.build:skip_test=True'",
    "tools.gnu:make_program": "Indicate path to make program",
    "tools.gnu:disable_flags": "Disable the automatic addition of flags to some build systems. List of possible values: ['arch', 'arch_link', 'libcxx', 'build_type', 'build_type_link', 'threads','cppstd', 'cstd']",
    "tools.gnu:define_libcxx11_abi": "Force definition of GLIBCXX_USE_CXX11_ABI=1 for libstdc++11",
    "tools.gnu:pkg_config": "Path to pkg-config executable used by PkgConfig build helper",
    "tools.gnu:build_triplet": "Custom build triplet to pass to Autotools scripts",
    "tools.gnu:host_triplet": "Custom host triplet to pass to Autotools scripts",
    "tools.gnu:extra_configure_args": "List of extra arguments to pass to configure when using AutotoolsToolchain and GnuToolchain",
    "tools.google.bazel:configs": "List of Bazel configurations to be used as 'bazel build --config=config1 ...'",
    "tools.google.bazel:bazelrc_path": "List of paths to bazelrc files to be used as 'bazel --bazelrc=rcpath1 ... build'",
    "tools.meson.mesontoolchain:backend": "Any Meson backend: ninja, vs, vs2010, vs2012, vs2013, vs2015, vs2017, vs2019, xcode",
    "tools.meson.mesontoolchain:extra_machine_files": "List of paths for any additional native/cross file references to be appended to the existing Conan ones",
    "tools.microsoft:winsdk_version": "Use this winsdk_version in vcvars",
    "tools.microsoft:msvc_update": "Force the specific update irrespective of compiler.update (CMakeToolchain and VCVars)",
    "tools.microsoft.msbuild:vs_version": "Defines the IDE version (15, 16, 17) when using the msvc compiler. Necessary if compiler.version specifies a toolset that is not the IDE default",
    "tools.microsoft.msbuild:max_cpu_count": "Argument for the /m when running msvc to build parallel projects",
    "tools.microsoft.msbuild:installation_path": "VS install path, to avoid auto-detect via vswhere, like C:/Program Files (x86)/Microsoft Visual Studio/2019/Community. Use empty string to disable",
    "tools.microsoft.msbuilddeps:exclude_code_analysis": "Suppress MSBuild code analysis for patterns",
    "tools.microsoft.msbuildtoolchain:compile_options": "Dictionary with MSBuild compiler options",
    "tools.microsoft.bash:subsystem": "The subsystem to be used when conanfile.win_bash==True. Possible values: msys2, msys, cygwin, wsl, sfu",
    "tools.microsoft.bash:path": "The path to the shell to run when conanfile.win_bash==True",
    "tools.microsoft.bash:active": "Set True only when Conan runs in a POSIX Bash (MSYS2/Cygwin) where Python's subprocess (shell=True) uses a POSIX-compatible shell (e.g., /bin/sh). Do not set when using Conan from cmd/PowerShell or with native Windows Python ('win32').",
    "tools.intel:installation_path": "Defines the Intel oneAPI installation root path",
    "tools.intel:setvars_args": "Custom arguments to be passed onto the setvars.sh|bat script from Intel oneAPI",
    "tools.system.package_manager:tool": "Default package manager tool: 'apk', 'apt-get', 'yum', 'dnf', 'brew', 'pacman', 'choco', 'zypper', 'pkg' or 'pkgutil'",
    "tools.system.package_manager:mode": "Mode for package_manager tools: 'check', 'report', 'report-installed' or 'install'",
    "tools.system.package_manager:sudo": "Use 'sudo' when invoking the package manager tools in Linux (False by default)",
    "tools.system.package_manager:sudo_askpass": "Use the '-A' argument if using sudo in Linux to invoke the system package manager (False by default)",
    "tools.system.pipenv:python_interpreter": "Path to the Python interpreter to be used to create the virtualenv",
    "tools.apple:sdk_path": "Path to the SDK to be used",
    "tools.apple:enable_bitcode": "(boolean) Enable/Disable Bitcode Apple Clang flags",
    "tools.apple:enable_arc": "(boolean) Enable/Disable ARC Apple Clang flags",
    "tools.apple:enable_visibility": "(boolean) Enable/Disable Visibility Apple Clang flags",
    "tools.env.virtualenv:powershell": "If specified, it generates PowerShell launchers (.ps1). Use this configuration setting the PowerShell executable you want to use (e.g., 'powershell.exe' or 'pwsh'). Setting it to True or False is deprecated as of Conan 2.11.0.",
    "tools.env:dotenv": "(Experimental) Generate dotenv environment files",
    "tools.env:deactivation_mode": "(Experimental) If 'function', generate a deactivate function instead of a script to unset the environment variables",
    # Compilers/Flags configurations
    "tools.build:compiler_executables": "Defines a Python dict-like with the compilers path to be used. Allowed keys {'c', 'cpp', 'cuda', 'objc', 'objcxx', 'rc', 'fortran', 'asm', 'hip', 'ispc'}",
    "tools.build:cxxflags": "List of extra CXX flags used by different toolchains like CMakeToolchain, AutotoolsToolchain and MesonToolchain",
    "tools.build:cflags": "List of extra C flags used by different toolchains like CMakeToolchain, AutotoolsToolchain and MesonToolchain",
    "tools.build:defines": "List of extra definition flags used by different toolchains like CMakeToolchain, AutotoolsToolchain and MesonToolchain",
    "tools.build:sharedlinkflags": "List of extra flags used by different toolchains like CMakeToolchain, AutotoolsToolchain and MesonToolchain",
    "tools.build:exelinkflags": "List of extra flags used by different toolchains like CMakeToolchain, AutotoolsToolchain and MesonToolchain",
    "tools.build:linker_scripts": "List of linker script files to pass to the linker used by different toolchains like CMakeToolchain, AutotoolsToolchain, and MesonToolchain",
    # Toolchain installation
    "tools.build:install_strip": "(boolean) Strip the binaries when installing them with CMake, Meson and Autotools",
    # Package ID composition
    "tools.info.package_id:confs": "List of existing configuration to be part of the package ID",
}

BUILT_IN_CONFS = {key: value for key, value in sorted(BUILT_IN_CONFS.items())}

CORE_CONF_PATTERN = re.compile(r"^(core\..+|core):.*")
TOOLS_CONF_PATTERN = re.compile(r"^(tools\..+|tools):.*")
USER_CONF_PATTERN = re.compile(r"^(user\..+|user):.*")


def _is_profile_module(module_name):
    # These are the modules that are propagated to profiles and user recipes
    return TOOLS_CONF_PATTERN.match(module_name) or USER_CONF_PATTERN.match(module_name)


# FIXME: Refactor all the next classes because they are mostly the same as
#        conan.tools.env.environment ones
class _ConfVarPlaceHolder:
    pass


class _ConfValue:

    def __init__(self, name, value, path=False, update=None):
        if name != name.lower():
            raise ConanException("Conf '{}' must be lowercase".format(name))
        self._name = name
        self._value = value
        self._value_type = type(value)
        if isinstance(value, (_PackageOption, SettingsItem)):
            raise ConanException(f"Invalid 'conf' type, please use Python types (int, str, ...)")
        self._path = path
        self._update = update

    def __repr__(self):
        return repr(self._value)

    @property
    def value(self):
        if self._value_type is list and _ConfVarPlaceHolder in self._value:
            v = self._value[:]
            v.remove(_ConfVarPlaceHolder)
            return v
        return self._value

    def copy(self):
        # Using copy for when self._value is a mutable list
        return _ConfValue(self._name, copy.copy(self._value), self._path, self._update)

    def dumps(self):
        if self._value is None:
            return "{}=!".format(self._name)  # unset
        elif self._value_type is list and _ConfVarPlaceHolder in self._value:
            v = self._value[:]
            v.remove(_ConfVarPlaceHolder)
            return "{}={}".format(self._name, v)
        else:
            return "{}={}".format(self._name, self._value)

    def serialize(self):
        if self._value is None:
            _value = "!"  # unset
        elif self._value_type is list and _ConfVarPlaceHolder in self._value:
            v = self._value[:]
            v.remove(_ConfVarPlaceHolder)
            _value = v
        else:
            _value = self._value
        return {self._name: _value}

    def update(self, value):
        assert self._value_type is dict, "Only dicts can be updated"
        assert isinstance(value, dict), "Only dicts can update"
        self._value.update(value)

    def remove(self, value):
        if self._value_type is list:
            self._value.remove(value)
        elif self._value_type is dict:
            self._value.pop(value, None)

    def append(self, value):
        if self._value_type is not list:
            raise ConanException("Only list-like values can append other values.")

        if isinstance(value, list):
            self._value.extend(value)
        else:
            if isinstance(value, (_PackageOption, SettingsItem)):
                raise ConanException(f"Invalid 'conf' type, please use Python types (int, str, ...)")
            self._value.append(value)

    def prepend(self, value):
        if self._value_type is not list:
            raise ConanException("Only list-like values can prepend other values.")

        if isinstance(value, list):
            self._value = value + self._value
        else:
            if isinstance(value, (_PackageOption, SettingsItem)):
                raise ConanException(f"Invalid 'conf' type, please use Python types (int, str, ...)")
            self._value.insert(0, value)

    def compose_conf_value(self, other):
        """
        self has precedence, the "other" will add/append if possible and not conflicting, but
        self mandates what to do. If self has define(), without placeholder, that will remain.
        :type other: _ConfValue
        """
        v_type = self._value_type
        o_type = other._value_type
        if v_type is list and o_type is list:
            try:
                index = self._value.index(_ConfVarPlaceHolder)
            except ValueError:  # It doesn't have placeholder
                pass
            else:
                new_value = self._value[:]  # do a copy
                new_value[index:index + 1] = other._value  # replace the placeholder
                self._value = new_value
        elif v_type is dict and o_type is dict:
            if self._update:
                # only if the current one is marked as "*=" update, otherwise it remains
                # as this is a "compose" operation, self has priority, it is the one updating
                new_value = other._value.copy()
                new_value.update(self._value)
                self._value = new_value
        elif issubclass(v_type, numbers.Number) and issubclass(o_type, numbers.Number):
            # They might be different kind of numbers, so skip the check below
            pass
        elif self._value is None or other._value is None:
            # It means any of those values were an "unset" so doing nothing because we don't
            # really know the original value type
            pass
        elif o_type != v_type:
            raise ConanException("It's not possible to compose {} values "
                                 "and {} ones.".format(v_type.__name__, o_type.__name__))
        # TODO: In case of any other object types?

    def set_relative_base_folder(self, folder):
        if not self._path:
            return
        if isinstance(self._value, list):
            self._value = [os.path.join(folder, v) if v != _ConfVarPlaceHolder else v
                           for v in self._value]
        if isinstance(self._value, dict):
            self._value = {k: os.path.join(folder, v) for k, v in self._value.items()}
        elif isinstance(self._value, str):
            self._value = os.path.join(folder, self._value)


class Conf:
    # Putting some default expressions to check that any value could be false
    boolean_false_expressions = ("0", '"0"', "false", '"false"', "off")
    boolean_true_expressions = ("1", '"1"', "true", '"true"', "on")

    def __init__(self):
        # It being ordered allows for Windows case-insensitive composition
        self._values = OrderedDict()  # {var_name: [] of values, including separators}

    def __bool__(self):
        return bool(self._values)

    def __repr__(self):
        return "Conf: " + repr(self._values)

    def __eq__(self, other):
        """
        :type other: Conf
        """
        return other._values == self._values

    def clear(self):
        self._values.clear()

    def validate(self):
        for conf in self._values:
            self._check_conf_name(conf)

    def items(self):
        # FIXME: Keeping backward compatibility
        for k, v in self._values.items():
            yield k, v.value

    def get(self, conf_name, default=None, check_type=None, choices=None):
        """
        Get all the values of the given configuration name.

        :param conf_name: Name of the configuration.
        :param default: Default value in case of conf does not have the conf_name key.
        :param check_type: Check the conf type(value) is the same as the given by this param.
                           There are two default smart conversions for bool and str types.
        :param choices: list of possible values this conf can have, if value not in it, errors.
        """
        # Skipping this check only the user.* configurations
        self._check_conf_name(conf_name)

        conf_value = self._values.get(conf_name)
        if conf_value:
            v = conf_value.value
            if choices is not None and v not in choices and v is not None:
                raise ConanException(f"Unknown value '{v}' for '{conf_name}'")
            # Some smart conversions
            if check_type is bool and not isinstance(v, bool):
                if str(v).lower() in Conf.boolean_false_expressions:
                    return False
                if str(v).lower() in Conf.boolean_true_expressions:
                    return True
                raise ConanException(f"[conf] {conf_name} must be a boolean-like object "
                                     f"(true/false, 1/0, on/off) and value '{v}' does not match it.")
            elif check_type is str and not isinstance(v, str):
                return str(v)
            elif v is None:  # value was unset
                return default
            elif (check_type is not None and not isinstance(v, check_type) or
                  check_type is int and isinstance(v, bool)):
                raise ConanException(f"[conf] {conf_name} must be a "
                                     f"{check_type.__name__}-like object. The value '{v}' "
                                     f"introduced is a {type(v).__name__} object")
            return v
        else:
            return default

    def pop(self, conf_name, default=None):
        """
        Remove the given configuration, returning its value.

        :param conf_name: Name of the configuration.
        :param default: Default value to return in case the configuration doesn't exist.
        :return:
        """
        value = self.get(conf_name, default=default)
        self._values.pop(conf_name, None)
        return value

    def show(self, fnpattern, pattern=""):
        return {key: self.get(key)
                for key in self._values.keys()
                if fnmatch.fnmatch(pattern + key, fnpattern)}

    def copy(self):
        c = Conf()
        c._values = OrderedDict((k, v.copy()) for k, v in self._values.items())
        return c

    def filter_core(self):
        c = Conf()
        c._values = OrderedDict((k, v.copy()) for k, v in self._values.items()
                                if not CORE_CONF_PATTERN.match(k))
        return c

    def dumps(self):
        """
        Returns a string with the format ``name=conf-value``
        """
        return "\n".join([v.dumps() for v in sorted(self._values.values(), key=lambda x: x._name)])

    def serialize(self):
        """
        Returns a dict-like object, e.g., ``{"tools.xxxx": "value1"}``
        """
        ret = {}
        for v in self._values.values():
            ret.update(v.serialize())
        return ret

    def define(self, name, value):
        """
        Define a value for the given configuration name.

        :param name: Name of the configuration.
        :param value: Value of the configuration.
        """
        self._values[name] = _ConfValue(name, value)

    def define_path(self, name, value):
        self._values[name] = _ConfValue(name, value, path=True)

    def unset(self, name):
        """
        Clears the variable, equivalent to a unset or set XXX=

        :param name: Name of the configuration.
        """
        self._values[name] = _ConfValue(name, None)

    def update(self, name, value):
        """
        Update the value to the given configuration name.

        :param name: Name of the configuration.
        :param value: Value of the configuration.
        """
        # Placeholder trick is not good for dict update, so we need to explicitly update=True
        conf_value = _ConfValue(name, {}, update=True)
        self._values.setdefault(name, conf_value).update(value)

    def update_path(self, name, value):
        conf_value = _ConfValue(name, {}, path=True, update=True)
        self._values.setdefault(name, conf_value).update(value)

    def append(self, name, value):
        """
        Append a value to the given configuration name.

        :param name: Name of the configuration.
        :param value: Value to append.
        """
        conf_value = _ConfValue(name, [_ConfVarPlaceHolder])
        self._values.setdefault(name, conf_value).append(value)

    def append_path(self, name, value):
        conf_value = _ConfValue(name, [_ConfVarPlaceHolder], path=True)
        self._values.setdefault(name, conf_value).append(value)

    def prepend(self, name, value):
        """
        Prepend a value to the given configuration name.

        :param name: Name of the configuration.
        :param value: Value to prepend.
        """
        conf_value = _ConfValue(name, [_ConfVarPlaceHolder])
        self._values.setdefault(name, conf_value).prepend(value)

    def prepend_path(self, name, value):
        conf_value = _ConfValue(name, [_ConfVarPlaceHolder], path=True)
        self._values.setdefault(name, conf_value).prepend(value)

    def remove(self, name, value):
        """
        Remove a value from the given configuration name.

        :param name: Name of the configuration.
        :param value: Value to remove.
        """
        conf_value = self._values.get(name)
        if conf_value:
            conf_value.remove(value)
        else:
            raise ConanException("Conf {} does not exist.".format(name))

    def compose_conf(self, other):
        """
        :param other: other has less priority than current one
        :type other: Conf
        """
        for k, v in other._values.items():
            existing = self._values.get(k)
            if existing is None:
                self._values[k] = v.copy()
            else:
                existing.compose_conf_value(v)
        return self

    def copy_conaninfo_conf(self):
        """
        Get a new `Conf()` object with all the configurations required by the consumer
        to be included in the final `ConanInfo().package_id()` computation. For instance, let's
        suppose that we have this Conan `profile`:

        ```
        ...
        [conf]
        tools.info.package_id:confs=["tools.build:cxxflags", "tools.build:cflags"]
        tools.build:cxxflags=["flag1xx"]
        tools.build:cflags=["flag1"]
        tools.build:defines=["DEF1"]
        ...

        Then, the resulting `Conf()` will have only these configuration lines:

        tools.build:cxxflags=["flag1xx"]
        tools.build:cflags=["flag1"]
        ```

        :return: a new `< Conf object >` with the configuration selected by `tools.info.package_id:confs`.
        """
        result = Conf()
        # Reading the list of all the configurations selected by the user to use for the package_id
        package_id_confs = self.get("tools.info.package_id:confs", default=[], check_type=list)
        for conf_name in package_id_confs:
            matching_confs = [c for c in self._values if re.match(conf_name, c)]
            for name in matching_confs:
                value = self.get(name)
                # Pruning any empty values, those should not affect package ID
                if value:
                    result.define(name, value)
        return result

    def set_relative_base_folder(self, folder):
        for v in self._values.values():
            v.set_relative_base_folder(folder)

    @staticmethod
    def _check_conf_name(conf):
        if conf.startswith("user"):
            if USER_CONF_PATTERN.match(conf) is None:
                raise ConanException(f"User conf '{conf}' invalid format, not 'user.org.group:conf'")
        elif conf not in BUILT_IN_CONFS:
            raise ConanException(f"[conf] '{conf}' does not exist in configuration list. "
                                 "Run 'conan config list' to see all the available confs.")


class ConfDefinition:
    # Order is important, "define" must be latest
    actions = (("+=", "append"), ("=+", "prepend"),
               ("=!", "unset"), ("*=", "update"), ("=", "define"))

    def __init__(self):
        self._pattern_confs = OrderedDict()

    def __repr__(self):
        return "ConfDefinition: " + repr(self._pattern_confs)

    def __bool__(self):
        return bool(self._pattern_confs)

    def get(self, conf_name, default=None, check_type=None, choices=None):
        """
        Get the value of the conf name requested and convert it to the [type]-like passed.
        """
        pattern, name = self._split_pattern_name(conf_name)
        return self._pattern_confs.get(pattern, Conf()).get(name, default=default,
                                                            check_type=check_type, choices=choices)

    def show(self, fnpattern):
        """
        Get the value of the confs that match the requested pattern
        """
        result = {}

        for patter_key, patter_conf in self._pattern_confs.items():
            if patter_key is None:
                patter_key = ""
            else:
                patter_key += ":"

            pattern_values = patter_conf.show(fnpattern, patter_key)
            result.update({patter_key + pattern_subkey: pattern_subvalue
                           for pattern_subkey, pattern_subvalue in pattern_values.items()})

        return result

    def pop(self, conf_name, default=None):
        """
        Remove the conf name passed.
        """
        pattern, name = self._split_pattern_name(conf_name)
        return self._pattern_confs.get(pattern, Conf()).pop(name, default=default)

    @staticmethod
    def _split_pattern_name(pattern_name):
        if pattern_name.count(":") >= 2:
            pattern, name = pattern_name.split(":", 1)
        else:
            pattern, name = None, pattern_name
        return pattern, name

    def get_conanfile_conf(self, ref, is_consumer=False):
        """ computes package-specific Conf
        it is only called when conanfile.buildenv is called
        the last one found in the profile file has top priority
        """
        result = Conf()
        for pattern, conf in self._pattern_confs.items():
            if pattern is None or ref_matches(ref, pattern, is_consumer):
                # Latest declared has priority, copy() necessary to not destroy data
                result = conf.copy().compose_conf(result)
        return result

    def update_conf_definition(self, other):
        """
        :type other: ConfDefinition
        :param other: The argument profile has priority/precedence over the current one.
        """
        for pattern, conf in other._pattern_confs.items():
            self._update_conf_definition(pattern, conf)

    def _update_conf_definition(self, pattern, conf):
        existing = self._pattern_confs.get(pattern)
        if existing:
            self._pattern_confs[pattern] = conf.compose_conf(existing)
        else:
            self._pattern_confs[pattern] = conf

    def rebase_conf_definition(self, global_conf):
        """
        for taking the new global.conf and composing with the profile [conf]
        :type global_conf: ConfDefinition
        """
        result = ConfDefinition()
        # Do not add ``core.xxx`` configuration to profiles
        for k, v in global_conf._pattern_confs.items():
            result._pattern_confs[k] = v.filter_core()
        result.update_conf_definition(self)
        self._pattern_confs = result._pattern_confs
        return

    def update(self, key, value, profile=False, method="define"):
        """
        Define/append/prepend/unset any Conf line
        >> update("tools.build:verbosity", "verbose")
        """
        pattern, name = self._split_pattern_name(key)

        if not _is_profile_module(name):
            if profile:
                raise ConanException("[conf] '{}' not allowed in profiles".format(key))
            if pattern is not None:
                raise ConanException("Conf '{}' cannot have a package pattern".format(key))

        # strip whitespaces before/after =
        # values are not strip() unless they are a path, to preserve potential whitespaces
        name = name.strip()

        # When loading from profile file, latest line has priority
        conf = Conf()
        if method == "unset":
            conf.unset(name)
        else:
            getattr(conf, method)(name, value)
        # Update
        self._update_conf_definition(pattern, conf)

    def dumps(self):
        result = []
        for pattern, conf in self._pattern_confs.items():
            if pattern is None:
                result.append(conf.dumps())
            else:
                result.append("\n".join("{}:{}".format(pattern, line) if line else ""
                                        for line in conf.dumps().splitlines()))
        if result:
            result.append("")
        return "\n".join(result)

    def serialize(self):
        result = {}
        for pattern, conf in self._pattern_confs.items():
            if pattern is None:
                result.update(conf.serialize())
            else:
                for k, v in conf.serialize().items():
                    result[f"{pattern}:{k}"] = v
        return result

    @staticmethod
    def _get_evaluated_value(_v):
        """
        Function to avoid eval() catching local variables
        """
        try:
            value = eval(_v)  # This destroys Windows path strings with backslash
        except:  # It means eval() failed because of a string without quotes
            value = _v.strip()
        else:
            if not isinstance(value, (numbers.Number, bool, dict, list, set, tuple)) \
                    and value is not None:
                # If it is quoted string we respect it as-is
                value = _v.strip()
        return value

    def loads(self, text, profile=False):
        self._pattern_confs = {}

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for op, method in ConfDefinition.actions:
                tokens = line.split(op, 1)
                if len(tokens) != 2:
                    continue
                pattern_name, value = tokens
                parsed_value = ConfDefinition._get_evaluated_value(value)
                self.update(pattern_name, parsed_value, profile=profile, method=method)
                break
            else:
                raise ConanException("Bad conf definition: {}".format(line))

    def validate(self):
        for conf in self._pattern_confs.values():
            conf.validate()

    def clear(self):
        self._pattern_confs.clear()


def load_global_conf(home_folder):
    home_paths = HomePaths(home_folder)
    global_conf_path = home_paths.global_conf_path
    new_config = ConfDefinition()
    if os.path.exists(global_conf_path):
        text = load(global_conf_path)
        distro = None
        if platform.system() in ["Linux", "FreeBSD"]:
            import distro
        template = Environment(loader=FileSystemLoader(home_folder)).from_string(text)
        home_folder = home_folder.replace("\\", "/")
        from conan import conan_version
        content = template.render({"platform": platform, "os": os, "distro": distro,
                                   "conan_version": conan_version,
                                   "conan_home_folder": home_folder,
                                   "detect_api": detect_api,
                                   "hashlib": hashlib})
        new_config.loads(content)
    else:  # creation of a blank global.conf file for user convenience
        default_global_conf = textwrap.dedent("""\
            # Core configuration (type 'conan config list' to list possible values)
            # e.g, for CI systems, to raise if user input would block
            # core:non_interactive = True
            # some tools.xxx config also possible, though generally better in profiles
            # tools.android:ndk_path = my/path/to/android/ndk
            """)
        save(global_conf_path, default_global_conf)
    return new_config
