from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main

conanfile_exe = '''from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


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
    exports_sources = "CMakeLists.txt", "src/*"

    def layout(self):
        cmake_layout(self)

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

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
'''

cmake_exe_v2 = """cmake_minimum_required(VERSION 3.15)
project({{name}} CXX)

{% if requires is defined -%}
{% for require in requires -%}
find_package({{as_name(require)}} CONFIG REQUIRED)
{% endfor %}
{%- endif %}

add_executable({{name}} src/{{name}}.cpp src/main.cpp)

{% if requires is defined -%}
{% for require in requires -%}
target_link_libraries({{name}} PRIVATE {{as_name(require)}}::{{as_name(require)}})
{% endfor %}
{%- endif %}

install(TARGETS {{name}} DESTINATION "."
        RUNTIME DESTINATION bin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        )
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

cmake_exe_files = {"conanfile.py": conanfile_exe,
                   "src/{{name}}.cpp": source_cpp,
                   "src/{{name}}.h": source_h,
                   "src/main.cpp": test_main,
                   "CMakeLists.txt": cmake_exe_v2,
                   "test_package/conanfile.py": test_conanfile_exe_v2
                   }
