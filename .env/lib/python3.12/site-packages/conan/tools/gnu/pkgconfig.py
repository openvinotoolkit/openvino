import textwrap
from io import StringIO

from conan.tools.build import cmd_args_to_string
from conan.tools.env import Environment
from conan.errors import ConanException


class PkgConfig:

    def __init__(self, conanfile, library, pkg_config_path=None):
        """

        :param conanfile: The current recipe object. Always use ``self``.
        :param library: The library which ``.pc`` file is to be parsed. It must exist in the pkg_config path.
        :param pkg_config_path:  If defined it will be prepended to ``PKG_CONFIG_PATH`` environment
               variable, so the execution finds the required files.
        """
        self._conanfile = conanfile
        self._library = library
        self._info = {}
        self._pkg_config_path = pkg_config_path
        self._variables = None

    def _parse_output(self, option):
        executable = self._conanfile.conf.get("tools.gnu:pkg_config", default="pkg-config")
        command = cmd_args_to_string([executable, '--' + option, self._library, '--print-errors'])

        env = Environment()
        if self._pkg_config_path:
            env.prepend_path("PKG_CONFIG_PATH", self._pkg_config_path)
        with env.vars(self._conanfile).apply():
            # This way we get the environment from ConanFile, from profile (default buildenv)
            output, err = StringIO(), StringIO()
            ret = self._conanfile.run(command, stdout=output, stderr=err, quiet=True,
                                      ignore_errors=True)
            if ret != 0:
                raise ConanException(f"PkgConfig failed. Command: {command}\n"
                                     f"    stdout:\n{textwrap.indent(output.getvalue(), '    ')}\n"
                                     f"    stderr:\n{textwrap.indent(err.getvalue(), '    ')}\n")
        value = output.getvalue().strip()
        return value

    def _get_option(self, option):
        if option not in self._info:
            self._info[option] = self._parse_output(option)
        return self._info[option]

    @property
    def includedirs(self):
        return [include[2:] for include in self._get_option('cflags-only-I').split()]

    @property
    def cflags(self):
        return [flag for flag in self._get_option('cflags-only-other').split()
                if not flag.startswith("-D")]

    @property
    def defines(self):
        return [flag[2:] for flag in self._get_option('cflags-only-other').split()
                if flag.startswith("-D")]

    @property
    def libdirs(self):
        return [lib[2:] for lib in self._get_option('libs-only-L').split()]

    @property
    def libs(self):
        return [lib[2:] for lib in self._get_option('libs-only-l').split()]

    @property
    def linkflags(self):
        return self._get_option('libs-only-other').split()

    @property
    def provides(self):
        return self._get_option('print-provides')

    @property
    def version(self):
        return self._get_option('modversion')

    @property
    def variables(self):
        if self._variables is None:
            variable_names = self._parse_output('print-variables').split()
            self._variables = {}
            for name in variable_names:
                self._variables[name] = self._parse_output('variable=%s' % name)
        return self._variables

    def fill_cpp_info(self, cpp_info, is_system=True, system_libs=None):
        """
        Method to fill a cpp_info object from the PkgConfig configuration

        :param cpp_info: Can be the global one (self.cpp_info) or a component one (self.components["foo"].cpp_info).
        :param is_system: If ``True``, all detected libraries will be assigned to ``cpp_info.system_libs``, and none to ``cpp_info.libs``.
        :param system_libs: If ``True``, all detected libraries will be assigned to ``cpp_info.system_libs``, and none to ``cpp_info.libs``.

        """
        if not self.provides:
            raise ConanException("PkgConfig error, '{}' files not available".format(self._library))
        self._conanfile.output.verbose(f"PkgConfig fill cpp_info for {self._library}")
        if is_system:
            cpp_info.system_libs = self.libs
        else:
            system_libs = system_libs or []
            cpp_info.libs = [lib for lib in self.libs if lib not in system_libs]
            cpp_info.system_libs = [lib for lib in self.libs if lib in system_libs]
        cpp_info.libdirs = self.libdirs
        cpp_info.sharedlinkflags = self.linkflags
        cpp_info.exelinkflags = self.linkflags
        cpp_info.defines = self.defines
        cpp_info.includedirs = self.includedirs
        cpp_info.cflags = self.cflags
        cpp_info.cxxflags = self.cflags
