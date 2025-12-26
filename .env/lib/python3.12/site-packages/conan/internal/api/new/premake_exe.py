from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main

conanfile_exe = """import os
from conan import ConanFile
from conan.tools.files import copy
from conan.tools.layout import basic_layout
from conan.tools.premake import PremakeDeps, PremakeToolchain, Premake


class {{package_name}}Recipe(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "application"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of {{ name }} package here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "premake5.lua", "src/*"

    def layout(self):
        basic_layout(self)

    def generate(self):
        deps = PremakeDeps(self)
        deps.generate()
        tc = PremakeToolchain(self)
        tc.generate()

    def build(self):
        premake = Premake(self)
        premake.configure()
        premake.build(workspace="{{name.capitalize()}}")

    def package(self):
        dest_bin = os.path.join(self.package_folder, "bin")
        build = os.path.join(self.build_folder, "bin")
        copy(self, "{{name}}", build, dest_bin, keep_path=False)
        copy(self, "{{name}}.exe", build, dest_bin, keep_path=False)


    {% if requires is defined -%}
    def requirements(self):
        {% for require in requires -%}
        self.requires("{{ require }}")
        {% endfor %}
    {%- endif %}

    {% if tool_requires is defined -%}
    def build_requirements(self):
        {% for require in tool_requires -%}
        self.tool_requires("{{ require }}")
        {% endfor %}
    {%- endif %}
"""

premake5 = """workspace "{{name.capitalize()}}"
   configurations { "Debug", "Release" }

project "{{name}}"
    kind "ConsoleApp"
    language "C++"
    files { "src/main.cpp", "src/{{name}}.cpp", "src/{{name}}.h" }

    filter "configurations:Debug"
       defines { "DEBUG" }
       symbols "On"

    filter "configurations:Release"
       defines { "NDEBUG" }
       optimize "On"
"""

test_conanfile_exe_v2 = """
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

premake_exe_files = {
    "conanfile.py": conanfile_exe,
    "src/{{name}}.cpp": source_cpp,
    "src/{{name}}.h": source_h,
    "src/main.cpp": test_main,
    "premake5.lua": premake5,
    "test_package/conanfile.py": test_conanfile_exe_v2,
}
