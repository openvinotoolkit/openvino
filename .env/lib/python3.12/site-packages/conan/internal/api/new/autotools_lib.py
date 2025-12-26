from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main


conanfile_sources_v2 = """
import os

from conan import ConanFile
from conan.tools.gnu import AutotoolsToolchain, Autotools
from conan.tools.layout import basic_layout
from conan.tools.apple import fix_apple_shared_install_name


class {{package_name}}Conan(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "library"
    win_bash = True

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of {package_name} here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    exports_sources = "configure.ac", "Makefile.am", "src/*"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        basic_layout(self)

    def generate(self):
        at_toolchain = AutotoolsToolchain(self)
        at_toolchain.generate()

    def build(self):
        autotools = Autotools(self)
        autotools.autoreconf()
        autotools.configure()
        autotools.make()

    def package(self):
        autotools = Autotools(self)
        autotools.install()
        fix_apple_shared_install_name(self)

    def package_info(self):
        self.cpp_info.libs = ["{{name}}"]
"""

configure_ac = """
AC_INIT([{{name}}], [{{version}}], [])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CXX
AM_PROG_AR
LT_INIT
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
"""

makefile_am = """
SUBDIRS = src
"""

makefile_am_lib = """
lib_LTLIBRARIES = lib{{name}}.la
lib{{name}}_la_SOURCES = {{name}}.cpp {{name}}.h
lib{{name}}_la_HEADERS = {{name}}.h
lib{{name}}_ladir = $(includedir)
"""

test_conanfile_v2 = """
import os

from conan import ConanFile
from conan.tools.gnu import AutotoolsToolchain, Autotools, AutotoolsDeps
from conan.tools.layout import basic_layout
from conan.tools.build import can_run


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "AutotoolsDeps", "AutotoolsToolchain"
    win_bash = True

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        autotools = Autotools(self)
        autotools.autoreconf()
        autotools.configure()
        autotools.make()

    def layout(self):
        basic_layout(self)

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.cpp.build.bindir, "main")
            self.run(cmd, env="conanrun")
"""

test_configure_ac = """
AC_INIT([main], [1.0], [])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CXX
AC_PROG_RANLIB
AM_PROG_AR
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
"""

test_makefile_am = """
bin_PROGRAMS = main
main_SOURCES = main.cpp
"""

autotools_lib_files = {"conanfile.py": conanfile_sources_v2,
                       "src/{{name}}.cpp": source_cpp,
                       "src/{{name}}.h": source_h,
                       "src/Makefile.am": makefile_am_lib,
                       "configure.ac": configure_ac,
                       "Makefile.am": makefile_am,
                       "test_package/conanfile.py": test_conanfile_v2,
                       "test_package/main.cpp": test_main,
                       "test_package/configure.ac": test_configure_ac,
                       "test_package/Makefile.am": test_makefile_am
                       }
