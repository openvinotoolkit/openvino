from conans import ConanFile, tools, CMake


class OpenvinoConan(ConanFile):
    name = "openvino"
    version = "2022.3"

    # metadata
    license = "Apache 2.0"
    author = "Intel Corporation"
    url = "https://github.com/openvinotoolkit/openvino"
    description = (
        "OpenVINO™ is an open-sorce toolkit for optimizing and deploying AI inference"
    )
    topics = ("deep-learning", "artificial-intelligence", "framework")

    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": True, "fPIC": True}
    generators = ["cmake", "CMakeDeps", "cmake_paths"]

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = (
        "cmake/*",
        "ngraph/*",
        "scripts/*",
        "src/*",
        "thirdparty/CMakeLists.txt",
        "thirdparty/ade/*",
        "thirdparty/cnpy/*",
        "thirdparty/fluid/*",
        "thirdparty/gflags/*",
        "thirdparty/itt_collector/*",
        "thirdparty/ittapi/*",
        "thirdparty/ocl/*",
        "thirdparty/ocv/*",
        "thirdparty/open_model_zoo/*",
        "thirdparty/protobuf/*",
        "thirdparty/xbyak/*",
        "tools/*",
        "samples/CMakeLists.txt",
        "docs/*",
        "licensing/*",
        "CMakeLists.txt",
        "LICENSE",
    )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def requirements(self):
        self.requires("nlohmann_json/3.11.2")
        self.requires("pugixml/1.13")
        self.requires("zlib/1.2.12")

    def build(self):
        cmake = CMake(self)
        cmake.verbose = True
        cmake.definitions["ENABLE_INTEL_CPU"] = "ON"
        cmake.definitions["ENABLE_INTEL_GPU"] = "OFF"
        cmake.definitions["ENABLE_INTEL_GNA"] = "OFF"
        cmake.definitions["ENABLE_INTEL_MYRIAD"] = "OFF"
        cmake.definitions["ENABLE_OPENCV"] = "OFF"
        cmake.definitions["ENABLE_TESTS"] = "OFF"
        cmake.definitions["ENABLE_BEH_TESTS"] = "OFF"
        cmake.definitions["ENABLE_FUNCTIONAL_TESTS"] = "OFF"
        cmake.definitions["ENABLE_PROFILING_ITT"] = "OFF"
        cmake.definitions["ENABLE_SAMPLES"] = "OFF"
        cmake.definitions["ENABLE_PYTHON"] = "OFF"
        cmake.definitions["ENABLE_CPPLINT"] = "OFF"
        cmake.definitions["ENABLE_NCC_STYLE"] = "OFF"
        cmake.definitions["ENABLE_OV_PADDLE_FRONTEND"] = "OFF"
        cmake.definitions["ENABLE_OV_TF_FRONTEND"] = "OFF"
        cmake.definitions["ENABLE_OV_ONNX_FRONTEND"] = "OFF"
        cmake.definitions["CMAKE_EXPORT_NO_PACKAGE_REGISTRY"] = "OFF"
        cmake.definitions["ENABLE_TEMPLATE"] = "OFF"
        cmake.definitions["ENABLE_INTEL_MYRIAD_COMMON"] = "OFF"
        cmake.definitions["ENABLE_COMPILE_TOOL"] = "OFF"
        cmake.configure()
        cmake.build()
        cmake.install()

    def package_info(self):
        self.cpp_info.libdirs = [
            "runtime/3rdparty/tbb_bind_2_5/lib",
            "tools/compile_tool",
            "runtime/lib/intel64",
        ]
        self.cpp_info.libs = ["openvino"]
        self.cpp_info.includedirs = [
            "runtime/include",
            "runtime/include/ie",
            "runtime/include/ngraph",
            "runtime/include/openvino",
        ]
