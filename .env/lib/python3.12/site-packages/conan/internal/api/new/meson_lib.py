from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main

conanfile_sources_v2 = """import os
from conan import ConanFile
from conan.tools.meson import MesonToolchain, Meson
from conan.tools.layout import basic_layout
from conan.tools.files import copy

class {{package_name}}Conan(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "library"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "meson.build", "src/*"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        basic_layout(self)

    def generate(self):
        tc = MesonToolchain(self)
        tc.generate()

    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def package(self):
        meson = Meson(self)
        meson.install()

    def package_info(self):
        self.cpp_info.libs = ["{{name}}"]
"""


test_conanfile_v2 = """import os
from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.meson import MesonToolchain, Meson
from conan.tools.layout import basic_layout


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "PkgConfigDeps", "MesonToolchain"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def layout(self):
        basic_layout(self)

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.cpp.build.bindir, "example")
            self.run(cmd, env="conanrun")
"""


_meson_build_test = """\
project('Test{{name}}', 'cpp')
{{name}} = dependency('{{name}}', version : '>=0.1')
executable('example', 'src/example.cpp', dependencies: {{name}})
"""


_meson_build = """\
project('{{name}} ', 'cpp')
library('{{name}}', 'src/{{name}}.cpp', install: true)
install_headers('src/{{name}}.h')
"""

meson_lib_files = {"conanfile.py": conanfile_sources_v2,
                   "src/{{name}}.cpp": source_cpp,
                   "src/{{name}}.h": source_h,
                   "meson.build": _meson_build,
                   "test_package/conanfile.py": test_conanfile_v2,
                   "test_package/src/example.cpp": test_main,
                   "test_package/meson.build": _meson_build_test}
