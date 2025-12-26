from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main

conanfile_sources = """import os
from conan import ConanFile
from conan.tools.files import copy
from conan.tools.layout import basic_layout
from conan.tools.premake import PremakeDeps, PremakeToolchain, Premake


class {{package_name}}Recipe(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "library"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of {{ name }} package here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}
    implements = ["auto_shared_fpic"]

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "premake5.lua", "src/*", "include/*"

    def layout(self):
        basic_layout(self)
    {% if requires is defined %}
    def requirements(self):
        {% for require in requires -%}
        self.requires("{{ require }}")
        {% endfor %}
    {%- endif %}
    {%- if tool_requires is defined %}
    def build_requirements(self):
        {% for require in tool_requires -%}
        self.tool_requires("{{ require }}")
        {% endfor %}
    {%- endif %}
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
        copy(self, "*.h", os.path.join(self.source_folder, "include"), os.path.join(self.package_folder, "include"))

        for pattern in ("*.lib", "*.a", "*.so*", "*.dylib"):
            copy(self, pattern, os.path.join(self.build_folder, "bin"), os.path.join(self.package_folder, "lib"))
        copy(self, "*.dll", os.path.join(self.build_folder, "bin"), os.path.join(self.package_folder, "bin"))

    def package_info(self):
        self.cpp_info.libs = ["{{name}}"]
"""

premake5 = """workspace "{{name.capitalize()}}"
    configurations { "Debug", "Release" }
    fatalwarnings {"All"}
    floatingpoint "Fast"
    includedirs { ".", "src", "include" }

project "{{name}}"
    cppdialect "C++17"

    -- To let conan take control over `kind` of the libraries, DO NOT SET `kind` (StaticLib or
    -- SharedLib) in `project` block.
    -- kind "<controlled by conan>"

    language "C++"
    files { "include/*.hpp", "include/*.h", "src/*.cpp" }

    filter "configurations:Debug"
       defines { "DEBUG" }
       symbols "On"

    filter "configurations:Release"
       defines { "NDEBUG" }
       optimize "On"

"""

test_conanfile = """import os

from conan import ConanFile
from conan.tools.layout import basic_layout
from conan.tools.premake import Premake
from conan.tools.build import can_run


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "PremakeDeps", "PremakeToolchain"

    def layout(self):
        basic_layout(self)

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        premake = Premake(self)
        premake.configure()
        premake.build(workspace="Example")

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.cpp.build.bindir, "bin", "example")
            self.run(cmd, env="conanrun")
"""

test_premake5 = """workspace "Example"
   configurations { "Debug", "Release" }

project "example"
   kind "ConsoleApp"
   language "C++"
   files { "src/example.cpp" }

   filter "configurations:Debug"
      defines { "DEBUG" }
      symbols "On"

   filter "configurations:Release"
      defines { "NDEBUG" }
      optimize "On"
"""


premake_lib_files = {
    "conanfile.py": conanfile_sources,
    "src/{{name}}.cpp": source_cpp,
    "include/{{name}}.h": source_h,
    "premake5.lua": premake5,
    "test_package/conanfile.py": test_conanfile,
    "test_package/src/example.cpp": test_main,
    "test_package/premake5.lua": test_premake5,
}
