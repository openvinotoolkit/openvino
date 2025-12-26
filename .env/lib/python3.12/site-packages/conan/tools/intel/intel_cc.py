"""
    Intel generator module for oneAPI Toolkits.

    For simplicity and clarity, Intel informally refers to some of the terms in this document,
    as listed below:

        - ICX - Intel oneAPI DPC++/C++ Compiler
        - ICC Classic - Intel C++ Compiler Classic
        - DPCPP - Intel oneAPI DPC++/C++ Compiler

    DPCPP is built upon ICX as the underlying C++ Compiler, therefore most of this information
    also applies to DPCPP.

    Intel oneAPI Toolkit (DPC++/C++ Compiler)
        - Versioning: https://software.intel.com/content/www/us/en/develop/articles/oneapi-toolkit-version-to-compiler-version-mapping.html
        - Compiler: https://software.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top.html
"""
import os
import platform
import textwrap

from conan.internal import check_duplicated_generator
from conan.errors import ConanException


def _is_using_intel_oneapi(compiler_version):
    """Check if the Intel compiler to be used belongs to Intel oneAPI

    Note: Intel oneAPI Toolkit first version is 2021.1
    """
    return int(compiler_version.split(".")[0]) >= 2021


class IntelCC:
    """Class that manages Intel oneAPI DPC++/C++/Classic Compilers vars generation"""

    filename = "conanintelsetvars"

    def __init__(self, conanfile):
        # Let's check the compatibility
        compiler_version = conanfile.settings.get_safe("compiler.version")
        mode = conanfile.settings.get_safe("compiler.mode")
        if _is_using_intel_oneapi(compiler_version):
            if mode != "classic" and conanfile.settings.get_safe("os") == "Darwin":
                raise ConanException(
                    'macOS* is not supported for the icx/icpx or dpcpp compilers. '
                    'Use the "classic" mode (icc compiler) instead.')
        else:
            # Do not support legacy versions
            raise ConanException("You have to use 'intel' compiler which is meant for legacy "
                                 "versions like Intel Parallel Studio XE.")
        # Private properties
        self._conanfile = conanfile
        self._settings = conanfile.settings
        self._compiler_version = compiler_version
        self._mode = mode
        self._out = conanfile.output
        # Public properties
        #: arch setting
        self.arch = conanfile.settings.get_safe("arch")

    @property
    def ms_toolset(self):
        """Get Microsoft Visual Studio Toolset depending on the mode selected"""
        if self._mode == "classic":
            # TODO: Get automatically the classic compiler version
            return "Intel C++ Compiler 19.2"
        elif self._mode == "icx":
            return "Intel C++ Compiler %s" % (self._compiler_version.split('.')[0])
        else:  # DPC++ compiler
            return "Intel(R) oneAPI DPC++ Compiler"

    def generate(self, scope="build"):
        """Generate the Conan Intel file to be loaded in build environment by default"""
        check_duplicated_generator(self, self._conanfile)
        intel_cc_path = self._conanfile.conf.get("tools.intel:installation_path")
        if intel_cc_path == "":
            return

        if platform.system() == "Windows" and not self._conanfile.win_bash:
            content = textwrap.dedent("""\
                @echo off
                {}
                """.format(self.command))
            filename = self.filename + '.bat'
        else:
            filename = self.filename + '.sh'
            content = self.command
        from conan.tools.env.environment import create_env_script
        create_env_script(self._conanfile, content, filename, scope)

    @property
    def installation_path(self):
        """Get the Intel oneAPI installation root path"""
        installation_path = self._conanfile.conf.get("tools.intel:installation_path")
        if not installation_path:
            raise ConanException("To use Intel oneAPI, specify a [conf] entry "
                                 "'tools.intel:installation_path' containing the path to the "
                                 "installation folder.")
        self._out.info("Got Intel oneAPI installation folder: %s" % installation_path)
        return installation_path

    @property
    def command(self):
        """
        The Intel oneAPI DPC++/C++ Compiler includes environment configuration scripts to
        configure your build and development environment variables:

        - On Linux, the file is a shell script called setvars.sh.
        - On Windows, the file is a batch file called setvars.bat.

        - Linux -> ``>> . /<install-dir>/setvars.sh <arg1> <arg2> … <argn><arg1> <arg2> … <argn>``
          The compiler environment script file accepts an optional target architecture
          argument <arg>:
          - intel64: Generate code and use libraries for Intel 64 architecture-based targets.
          - ia32: Generate code and use libraries for IA-32 architecture-based targets.

        - Windows -> ``>> call <install-dir>\\setvars.bat [<arg1>] [<arg2>]``
          Where <arg1> is optional and can be one of the following:
          - intel64: Generate code and use libraries for Intel 64 architecture (host and target).
          - ia32: Generate code and use libraries for IA-32 architecture (host and target).

          With the dpcpp compiler, <arg1> is intel64 by default.

          The <arg2> is optional. If specified, it is one of the following:
          - vs2019: Microsoft Visual Studio* 2019
          - vs2017: Microsoft Visual Studio 2017

        :return: `str` setvars.sh|bat command to be run
        """
        # Let's check if user wants to use some custom arguments to run the setvars script
        command_args = self._conanfile.conf.get("tools.intel:setvars_args", default="")
        system = platform.system()
        svars = "setvars.bat" if system == "Windows" else "setvars.sh"
        command = '"%s"' % os.path.join(self.installation_path, svars)
        if system == "Windows":
            command = "call " + command
        else:
            command = ". " + command  # dot is more portable than source
        # If user has passed custom arguments
        if command_args:
            command += " %s" % command_args
            return command
        # Add architecture argument
        if self.arch == "x86_64":
            command += " intel64"
        elif self.arch == "x86":
            command += " ia32"
        else:
            raise ConanException("don't know how to call %s for %s" % (svars, self.arch))

        return command
