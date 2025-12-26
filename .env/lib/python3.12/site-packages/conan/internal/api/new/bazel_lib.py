from conan.internal.api.new.cmake_lib import source_cpp, source_h, test_main

conanfile_sources_v2 = """
import os
from conan import ConanFile
from conan.tools.google import Bazel, bazel_layout
from conan.tools.files import copy

class {{package_name}}Recipe(ConanFile):
    name = "{{name}}"
    version = "{{version}}"
    package_type = "library"

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "main/*", "WORKSPACE", ".bazelrc"
    generators = "BazelToolchain"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def layout(self):
        bazel_layout(self)

    def build(self):
        from conan.api.output import ConanOutput
        ConanOutput().warning("This is the template for Bazel 6.x version, "
                              "but it will be overridden by the 'bazel_7_lib' template "
                              "(Bazel >= 7.1 compatible).", warn_tag="deprecated")
        bazel = Bazel(self)
        # On Linux platforms, Bazel creates both shared and static libraries by default, and
        # it is getting naming conflicts if we use the cc_shared_library rule
        if self.options.shared and self.settings.os != "Linux":
            # We need to add '--experimental_cc_shared_library' because the project uses
            # cc_shared_library to create shared libraries
            bazel.build(args=["--experimental_cc_shared_library"], target="//main:{{name}}_shared")
        else:
            bazel.build(target="//main:{{name}}")

    def package(self):
        dest_lib = os.path.join(self.package_folder, "lib")
        dest_bin = os.path.join(self.package_folder, "bin")
        build = os.path.join(self.build_folder, "bazel-bin", "main")
        copy(self, "*.so", build, dest_lib, keep_path=False)
        copy(self, "*.dll", build, dest_bin, keep_path=False)
        copy(self, "*.dylib", build, dest_lib, keep_path=False)
        if self.settings.os == "Linux" and self.options.get_safe("fPIC"):
            copy(self, "*.pic.a", build, dest_lib, keep_path=False)
        else:
            copy(self, "*.a", build, dest_lib, keep_path=False)
        copy(self, "*.lib", build, dest_lib, keep_path=False)
        copy(self, "{{name}}.h", os.path.join(self.source_folder, "main"),
             os.path.join(self.package_folder, "include"), keep_path=False)

    def package_info(self):
        if self.options.shared and self.settings.os != "Linux":
            self.cpp_info.libs = ["{{name}}_shared"]
        else:
            self.cpp_info.libs = ["{{name}}"]
"""


test_conanfile_v2 = """import os
from conan import ConanFile
from conan.tools.google import Bazel, bazel_layout
from conan.tools.build import can_run


class {{package_name}}TestConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "BazelToolchain", "BazelDeps"

    def requirements(self):
        self.requires(self.tested_reference_str)

    def build(self):
        bazel = Bazel(self)
        bazel.build()

    def layout(self):
        bazel_layout(self)

    def test(self):
        if can_run(self):
            cmd = os.path.join(self.cpp.build.bindir, "main", "example")
            self.run(cmd, env="conanrun")
"""


_bazel_build_test = """\
cc_binary(
    name = "example",
    srcs = ["example.cpp"],
    deps = [
        "@{{name}}//:{{name}}",
    ],
)
"""

_bazel_build = """\
cc_library(
    name = "{{name}}",
    srcs = ["{{name}}.cpp"],
    hdrs = ["{{name}}.h"],
)
"""

_bazel_build_shared = """
cc_shared_library(
    name = "{{name}}_shared",
    shared_lib_name = "lib{{name}}_shared.%s",
    deps = [":{{name}}"],
)
"""

_bazel_workspace = " "  # Important not empty, so template doesn't discard it
_bazel_rc = """\
{% if output_root_dir is defined %}startup --output_user_root={{output_root_dir}}{% endif %}
"""
_test_bazel_workspace = """
load("@//conan:dependencies.bzl", "load_conan_dependencies")
load_conan_dependencies()
"""


def _get_bazel_build():
    import platform
    os_ = platform.system()
    ret = _bazel_build
    if os_ != "Linux":
        ret += _bazel_build_shared % ("dylib" if os_ == "Darwin" else "dll")
    return ret


bazel_lib_files = {"conanfile.py": conanfile_sources_v2,
                   "main/{{name}}.cpp": source_cpp,
                   "main/{{name}}.h": source_h,
                   "main/BUILD": _get_bazel_build(),
                   "WORKSPACE": _bazel_workspace,
                   ".bazelrc": _bazel_rc,
                   "test_package/conanfile.py": test_conanfile_v2,
                   "test_package/main/example.cpp": test_main,
                   "test_package/main/BUILD": _bazel_build_test,
                   "test_package/WORKSPACE": _test_bazel_workspace,
                   "test_package/.bazelrc": _bazel_rc}
