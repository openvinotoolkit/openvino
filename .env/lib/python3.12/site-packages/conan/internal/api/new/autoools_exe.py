from conan.internal.api.new.autotools_lib import makefile_am
from conan.internal.api.new.cmake_lib import source_cpp, test_main, source_h

conanfile_exe = """\
import os

from conan import ConanFile
from conan.tools.gnu import AutotoolsToolchain, Autotools, AutotoolsDeps
from conan.tools.layout import basic_layout
from conan.tools.files import chdir


class {{package_name}}Conan(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "application"
    win_bash = True

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of {package_name} here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "configure.ac", "Makefile.am", "src/*"

    {% if requires is defined -%}
    def requirements(self):
        {% for require in requires -%}
        self.requires("{{ require }}")
        {% endfor %}
    {%- endif %}

    def layout(self):
        basic_layout(self)

    def generate(self):
        deps = AutotoolsDeps(self)
        deps.generate()
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
"""
makefile_am_exe = """\
bin_PROGRAMS = {{name}}
{{name}}_SOURCES = main.cpp {{name}}.cpp
"""

test_conanfile_exe_v2 = """\
import os
from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.layout import basic_layout


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def layout(self):
        basic_layout(self)

    def test(self):
        if can_run(self):
            self.run("{{name}}", env="conanrun")
"""


configure_ac = """\
AC_INIT([{{name}}], [{{version}}], [])
AM_INIT_AUTOMAKE([-Wall -Werror foreign])
AC_PROG_CXX
AC_PROG_RANLIB
AM_PROG_AR
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
"""


autotools_exe_files = {"conanfile.py": conanfile_exe,
                       "src/{{name}}.cpp": source_cpp,
                       "src/{{name}}.h": source_h,
                       "src/main.cpp": test_main,
                       "configure.ac": configure_ac,
                       "Makefile.am": makefile_am,
                       "src/Makefile.am": makefile_am_exe,
                       "test_package/conanfile.py": test_conanfile_exe_v2
                       }
