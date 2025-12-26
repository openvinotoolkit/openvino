from conan.internal.api.new.msbuild_lib import vcxproj, sln_file

test_main = """#include "{{name}}.h"

int main() {
    {{name}}();
}
"""


conanfile_exe = """import os

from conan import ConanFile
from conan.tools.microsoft import MSBuildToolchain, MSBuild, vs_layout
from conan.tools.files import copy


class {{package_name}}Conan(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "application"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "{{name}}.sln", "{{name}}.vcxproj", "src/*"

    def layout(self):
        vs_layout(self)

    def generate(self):
        tc = MSBuildToolchain(self)
        tc.generate()

    def build(self):
        msbuild = MSBuild(self)
        msbuild.build("{{name}}.sln")

    def package(self):
        copy(self, "*{{name}}.exe", src=self.build_folder,
             dst=os.path.join(self.package_folder, "bin"), keep_path=False)
"""


test_conanfile_exe_v2 = """import os
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

test_exe = r"""#include <iostream>

int main() {
    #ifdef NDEBUG
    std::cout << "{{name}}/{{version}}: Hello World Release!\n";
    #else
    std::cout << "{{name}}/{{version}}: Hello World Debug!\n";
    #endif
}
"""


msbuild_exe_files = {"conanfile.py": conanfile_exe,
                     "src/{{name}}.cpp": test_exe,
                     "{{name}}.sln": sln_file.replace("test_", ""),
                     "{{name}}.vcxproj": vcxproj.replace("TYPE_PLACEHOLDER", "Application")
                                                .replace("DEPENDENCIES", "").replace("test_", ""),
                     "test_package/conanfile.py": test_conanfile_exe_v2
                     }
