conanfile_sources_v2 = '''from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps


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

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)
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

    def package_info(self):
        self.cpp_info.libs = ["{{name}}"]

'''

cmake_v2 = """cmake_minimum_required(VERSION 3.15)
project({{name}} CXX)

{% if requires is defined -%}
{% for require in requires -%}
find_package({{as_name(require)}} CONFIG REQUIRED)
{% endfor %}
{%- endif %}


add_library({{name}} src/{{name}}.cpp)
target_include_directories({{name}} PUBLIC include)

{% if requires is defined -%}
{% for require in requires -%}
target_link_libraries({{name}} PRIVATE {{as_name(require)}}::{{as_name(require)}})
{% endfor %}
{%- endif %}

set_target_properties({{name}} PROPERTIES PUBLIC_HEADER "include/{{name}}.h")
install(TARGETS {{name}})
"""

source_h = """#pragma once

#include <vector>
#include <string>

{% set define_name = package_name.upper() %}
#ifdef _WIN32
  #define {{define_name}}_EXPORT __declspec(dllexport)
#else
  #define {{define_name}}_EXPORT
#endif

{{define_name}}_EXPORT void {{package_name}}();
{{define_name}}_EXPORT void {{package_name}}_print_vector(const std::vector<std::string> &strings);
"""

source_cpp = r"""#include <iostream>
#include "{{name}}.h"
{% if requires is defined -%}
{% for require in requires -%}
#include "{{ as_name(require) }}.h"
{% endfor %}
{%- endif %}


void {{package_name}}(){
    {% if requires is defined -%}
    {% for require in requires -%}
    {{ as_name(require).replace(".", "_") }}();
    {% endfor %}
    {%- endif %}

    #ifdef NDEBUG
    std::cout << "{{name}}/{{version}}: Hello World Release!\n";
    #else
    std::cout << "{{name}}/{{version}}: Hello World Debug!\n";
    #endif

    // ARCHITECTURES
    #ifdef _M_X64
    std::cout << "  {{name}}/{{version}}: _M_X64 defined\n";
    #endif

    #ifdef _M_IX86
    std::cout << "  {{name}}/{{version}}: _M_IX86 defined\n";
    #endif

    #ifdef _M_ARM64
    std::cout << "  {{name}}/{{version}}: _M_ARM64 defined\n";
    #endif

    #if __i386__
    std::cout << "  {{name}}/{{version}}: __i386__ defined\n";
    #endif

    #if __x86_64__
    std::cout << "  {{name}}/{{version}}: __x86_64__ defined\n";
    #endif

    #if __aarch64__
    std::cout << "  {{name}}/{{version}}: __aarch64__ defined\n";
    #endif

    // Libstdc++
    #if defined _GLIBCXX_USE_CXX11_ABI
    std::cout << "  {{name}}/{{version}}: _GLIBCXX_USE_CXX11_ABI "<< _GLIBCXX_USE_CXX11_ABI << "\n";
    #endif

    // MSVC runtime
    #if defined(_DEBUG)
        #if defined(_MT) && defined(_DLL)
        std::cout << "  {{name}}/{{version}}: MSVC runtime: MultiThreadedDebugDLL\n";
        #elif defined(_MT)
        std::cout << "  {{name}}/{{version}}: MSVC runtime: MultiThreadedDebug\n";
        #endif
    #else
        #if defined(_MT) && defined(_DLL)
        std::cout << "  {{name}}/{{version}}: MSVC runtime: MultiThreadedDLL\n";
        #elif defined(_MT)
        std::cout << "  {{name}}/{{version}}: MSVC runtime: MultiThreaded\n";
        #endif
    #endif

    // COMPILER VERSIONS
    #if _MSC_VER
    std::cout << "  {{name}}/{{version}}: _MSC_VER" << _MSC_VER<< "\n";
    #endif

    #if _MSVC_LANG
    std::cout << "  {{name}}/{{version}}: _MSVC_LANG" << _MSVC_LANG<< "\n";
    #endif

    #if __cplusplus
    std::cout << "  {{name}}/{{version}}: __cplusplus" << __cplusplus<< "\n";
    #endif

    #if __INTEL_COMPILER
    std::cout << "  {{name}}/{{version}}: __INTEL_COMPILER" << __INTEL_COMPILER<< "\n";
    #endif

    #if __GNUC__
    std::cout << "  {{name}}/{{version}}: __GNUC__" << __GNUC__<< "\n";
    #endif

    #if __GNUC_MINOR__
    std::cout << "  {{name}}/{{version}}: __GNUC_MINOR__" << __GNUC_MINOR__<< "\n";
    #endif

    #if __clang_major__
    std::cout << "  {{name}}/{{version}}: __clang_major__" << __clang_major__<< "\n";
    #endif

    #if __clang_minor__
    std::cout << "  {{name}}/{{version}}: __clang_minor__" << __clang_minor__<< "\n";
    #endif

    #if __apple_build_version__
    std::cout << "  {{name}}/{{version}}: __apple_build_version__" << __apple_build_version__<< "\n";
    #endif

    // SUBSYSTEMS

    #if __MSYS__
    std::cout << "  {{name}}/{{version}}: __MSYS__" << __MSYS__<< "\n";
    #endif

    #if __MINGW32__
    std::cout << "  {{name}}/{{version}}: __MINGW32__" << __MINGW32__<< "\n";
    #endif

    #if __MINGW64__
    std::cout << "  {{name}}/{{version}}: __MINGW64__" << __MINGW64__<< "\n";
    #endif

    #if __CYGWIN__
    std::cout << "  {{name}}/{{version}}: __CYGWIN__" << __CYGWIN__<< "\n";
    #endif
}

void {{package_name}}_print_vector(const std::vector<std::string> &strings) {
    for(std::vector<std::string>::const_iterator it = strings.begin(); it != strings.end(); ++it) {
        std::cout << "{{package_name}}/{{version}} " << *it << std::endl;
    }
}
"""


test_conanfile_v2 = """import os

from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.tools.build import can_run


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def layout(self):
        cmake_layout(self)

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.cpp.build.bindir, "example")
            self.run(cmd, env="conanrun")
"""

test_cmake_v2 = """cmake_minimum_required(VERSION 3.15)
project(PackageTest CXX)

find_package({{name}} CONFIG REQUIRED)

add_executable(example src/example.cpp)
target_link_libraries(example {{name}}::{{name}})
"""


test_main = """#include "{{name}}.h"
#include <vector>
#include <string>

int main() {
    {{package_name}}();

    std::vector<std::string> vec;
    vec.push_back("test_package");

    {{package_name}}_print_vector(vec);
}
"""

cmake_lib_files = {"conanfile.py": conanfile_sources_v2,
                   "src/{{name}}.cpp": source_cpp,
                   "include/{{name}}.h": source_h,
                   "CMakeLists.txt": cmake_v2,
                   "test_package/conanfile.py": test_conanfile_v2,
                   "test_package/src/example.cpp": test_main,
                   "test_package/CMakeLists.txt": test_cmake_v2}
