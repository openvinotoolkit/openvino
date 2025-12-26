from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main

conanfile_exe = """from conan import ConanFile
from conan.tools.meson import MesonToolchain, Meson
from conan.tools.gnu import PkgConfigDeps


class {{package_name}}Conan(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "application"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "meson.build", "src/*"

    {% if requires is defined -%}
    def requirements(self):
        {% for require in requires -%}
        self.requires("{{ require }}")
        {% endfor %}
    {%- endif %}

    def layout(self):
        self.folders.build = "build"

    def generate(self):
        deps = PkgConfigDeps(self)
        deps.generate()
        tc = MesonToolchain(self)
        tc.generate()

    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def package(self):
        meson = Meson(self)
        meson.install()
"""

test_conanfile_exe_v2 = """import os
from conan import ConanFile
from conan.tools.build import can_run


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def test(self):
        if can_run(self):
            self.run("{{name}}", env="conanrun")
"""

_meson_build_exe = """\
project('{{name}} ', 'cpp')
{% if requires is defined -%}
cxx = meson.get_compiler('cpp')
{% for require in requires -%}
{{as_name(require)}} = dependency('{{as_name(require)}}', required: true)
{% endfor %}
{%- endif %}
{% if requires is defined -%}
executable('{{name}}', 'src/{{name}}.cpp', 'src/main.cpp', install: true,
           dependencies:  {{ names(requires) }} )
{% else %}
executable('{{name}}', 'src/{{name}}.cpp', 'src/main.cpp', install: true)
{% endif %}
"""

meson_exe_files = {"conanfile.py": conanfile_exe,
                   "src/{{name}}.cpp": source_cpp,
                   "src/{{name}}.h": source_h,
                   "src/main.cpp": test_main,
                   "meson.build": _meson_build_exe,
                   "test_package/conanfile.py": test_conanfile_exe_v2
                   }
