# Build Plugin Using CMake* {#openvino_docs_ie_plugin_dg_plugin_build}

Inference Engine build infrastructure provides the Inference Engine Developer Package for plugin development.

Inference Engine Developer Package
------------------------

To automatically generate the Inference Engine Developer Package, run the `cmake` tool during a OpenVINO build:

```bash
$ mkdir openvino-release-build
$ cd openvino-release-build
$ cmake -DCMAKE_BUILD_TYPE=Release ../openvino 
```

Once the commands above are executed, the Inference Engine Developer Package is generated in the `openvino-release-build` folder. It consists of several files:
 - `InferenceEngineDeveloperPackageConfig.cmake` - the main CMake script which imports targets and provides compilation flags and CMake options.
 - `InferenceEngineDeveloperPackageConfig-version.cmake` - a file with a package version.
 - `targets_developer.cmake` - an automatically generated file which contains all targets exported from the OpenVINO build tree. This file is included by `InferenceEngineDeveloperPackageConfig.cmake` to import the following targets:
   - Libraries for plugin development:
       * `IE::ngraph` - shared nGraph library
       * `IE::inference_engine` - shared Inference Engine library
       * `IE::inference_engine_transformations` - shared library with Inference Engine ngraph-based Transformations
       * `IE::openvino_gapi_preproc` - shared library with Inference Engine preprocessing plugin
       * `IE::inference_engine_plugin_api` - interface library with Inference Engine Plugin API headers
       * `IE::inference_engine_lp_transformations` - shared library with low-precision transformations
       * `IE::pugixml` - static Pugixml library
       * `IE::xbyak` - interface library with Xbyak headers
       * `IE::itt` - static library with tools for performance measurement using Intel ITT
   - Libraries for tests development:
       * `IE::gtest`, `IE::gtest_main`, `IE::gmock` - Google Tests framework libraries
       * `IE::commonTestUtils` - static library with common tests utilities 
       * `IE::funcTestUtils` - static library with functional tests utilities 
       * `IE::unitTestUtils` - static library with unit tests utilities 
       * `IE::ngraphFunctions` - static library with the set of `ngraph::Function` builders
       * `IE::funcSharedTests` - static library with common functional tests

> **NOTE**: it's enough just to run `cmake --build . --target ie_dev_targets` command to build only targets from the
> Inference Engine Developer package.

Build Plugin using Inference Engine Developer Package
------------------------

To build a plugin source tree using the Inference Engine Developer Package, run the commands below:

```cmake
$ mkdir template-plugin-release-build
$ cd template-plugin-release-build
$ cmake -DInferenceEngineDeveloperPackage_DIR=../openvino-release-build ../template-plugin
```

A common plugin consists of the following components:

1. Plugin code in the `src` folder
2. Code of tests in the `tests` folder

To build a plugin and its tests, run the following CMake scripts:

- Root `CMakeLists.txt`, which finds the Inference Engine Developer Package using the `find_package` CMake command and adds the `src` and `tests` subdirectories with plugin sources and their tests respectively:

```cmake
cmake_minimum_required(VERSION 3.13)

project(OpenVINOTemplatePlugin)

set(TEMPLATE_PLUGIN_SOURCE_DIR ${OpenVINOTemplatePlugin_SOURCE_DIR})

find_package(OpenVINODeveloperPackage REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()

    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()
```

> **NOTE**: The default values of the `ENABLE_TESTS`, `ENABLE_FUNCTIONAL_TESTS` options are shared via the Inference Engine Developer Package and they are the same as for the main OpenVINO build tree. You can override them during plugin build using the command below:

    ```bash
    $ cmake -DENABLE_FUNCTIONAL_TESTS=OFF -DInferenceEngineDeveloperPackage_DIR=../openvino-release-build ../template-plugin
    ``` 

- `src/CMakeLists.txt` to build a plugin shared library from sources:

@snippet template/src/CMakeLists.txt cmake:plugin

> **NOTE**: `IE::inference_engine` target is imported from the Inference Engine Developer Package.

- `tests/functional/CMakeLists.txt` to build a set of functional plugin tests:

@snippet template/tests/functional/CMakeLists.txt cmake:functional_tests

> **NOTE**: The `IE::funcSharedTests` static library with common functional Inference Engine Plugin tests is imported via the Inference Engine Developer Package.
