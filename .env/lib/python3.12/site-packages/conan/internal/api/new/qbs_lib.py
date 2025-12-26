from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main


conanfile_sources = '''
import os

from conan import ConanFile
from conan.tools.qbs import Qbs


class {{package_name}}Recipe(ConanFile):
    name = "{{name}}"
    version = "{{version}}"

    exports_sources = "*.cpp", "*.h", "*.qbs"
    settings = "os", "compiler", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}

    def build(self):
        qbs = Qbs(self)
        qbs_config = {"products.{{name}}.isShared": "true" if self.options.shared else "false"}
        qbs.add_configuration("default", qbs_config)
        qbs.resolve()
        qbs.build()

    def package(self):
        qbs = Qbs(self)
        qbs.install()

    def package_info(self):
        self.cpp_info.libs = ["{{name}}"]
'''

qbs_lib_file = '''
    Library {
        property bool isShared: true
        name: "{{name}}"
        type: isShared ? "dynamiclibrary" : "staticlibrary"
        files: [ "{{name}}.cpp" ]
        Group {
            name: "headers"
            files: [ "{{name}}.h" ]
            qbs.install: true
            qbs.installDir: "include"
        }
        Depends { name: "cpp" }
        Depends { name: "bundle" }
        bundle.isBundle: false
        install: true
        qbs.installPrefix: ""
    }
'''

test_conanfile_v2 = """import os

from conan import ConanFile
from conan.tools.build import can_run
from conan.tools.qbs import Qbs
from conan.tools.build import cmd_args_to_string

class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "PkgConfigDeps"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        qbs = Qbs(self)
        qbs.resolve()
        qbs.build()
        qbs.install()

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.package_folder, "bin", "example")
            self.run(cmd_args_to_string([cmd]), env="conanrun")
"""

qbs_test_file = '''
    Application {
        name: "example"
        consoleApplication: true
        files: [ "example.cpp" ]
        Depends { name: "cpp" }
        install: true
        qbs.installPrefix: ""
        // external dependency via pkg-config
        qbsModuleProviders: ["qbspkgconfig"]
        moduleProviders.qbspkgconfig.libDirs: path
        Depends { name: "{{name}}" }
    }
'''

qbs_lib_files = {"conanfile.py": conanfile_sources,
                 "{{name}}.qbs": qbs_lib_file,
                 "{{name}}.cpp": source_cpp,
                 "{{name}}.h": source_h,
                 "test_package/conanfile.py": test_conanfile_v2,
                 "test_package/example.cpp": test_main,
                 "test_package/example.qbs": qbs_test_file}
