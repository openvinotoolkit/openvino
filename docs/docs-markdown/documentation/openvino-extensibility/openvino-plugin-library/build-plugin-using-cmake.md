---
sidebar_position: 8
sidebar_label: "Build Plugin Using CMake"
description: "Learn how to build a plugin using CMake and OpenVINO Developer Package."
---

# Build Plugin Using CMake

OpenVINO build infrastructure provides the OpenVINO Developer Package for plugin development.

## OpenVINO Developer Package

To automatically generate the OpenVINO Developer Package, run the `cmake` tool during a OpenVINO build:

```bash
$ mkdir openvino-release-build
$ cd openvino-release-build
$ cmake -DCMAKE_BUILD_TYPE=Release ../openvino
```

Once the commands above are executed, the OpenVINO Developer Package is generated in the `openvino-release-build` folder. It consists of several files:

- `OpenVINODeveloperPackageConfig.cmake` - the main CMake script which imports targets and provides compilation flags and CMake options.
- `OpenVINODeveloperPackageConfig-version.cmake` - a file with a package version.
- `targets_developer.cmake` - an automatically generated file which contains all targets exported from the OpenVINO build tree. This file is included by `OpenVINODeveloperPackageConfig.cmake` to import the following targets:
  
  - Libraries for plugin development:
    
    - `openvino::runtime` - shared OpenVINO library
    - `openvino::runtime::dev` - interface library with OpenVINO Developer API
    - `openvino::pugixml` - static Pugixml library
    - `openvino::xbyak` - interface library with Xbyak headers
    - `openvino::itt` - static library with tools for performance measurement using Intel ITT
  - Libraries for tests development:
    
    - `openvino::gtest`, `openvino::gtest_main`, `openvino::gmock` - Google Tests framework libraries
    - `openvino::common_test_utils` - static library with common tests utilities
    - `openvino::func_test_utils` - static library with functional tests utilities
    - `openvino::unit_test_utils` - static library with unit tests utilities
    - `openvino::ov_models` - static library with the set of `ov::Model` builders
    - `openvino::funcSharedTests` - static library with common functional tests

:::note
It's enough just to run `cmake --build . --target ov_dev_targets` command to build only targets from the OpenVINO Developer package.
:::

## Build Plugin using OpenVINO Developer Package

To build a plugin source tree using the OpenVINO Developer Package, run the commands below:

```bash
$ mkdir template-plugin-release-build
$ cd template-plugin-release-build
$ cmake -DOpenVINODeveloperPackage_DIR=../openvino-release-build ../template-plugin
```

A common plugin consists of the following components:

1. Plugin code in the `src` folder
2. Code of tests in the `tests` folder

To build a plugin and its tests, run the following CMake scripts:

- Root `CMakeLists.txt`, which finds the OpenVINO Developer Package using the `find_package` CMake command and adds the `src` and `tests` subdirectories with plugin sources and their tests respectively:

```cpp
cmake_minimum_required(VERSION 3.13)

project(OpenVINOTemplatePlugin)

find_package(OpenVINODeveloperPackage REQUIRED)

ov_option(ENABLE_TEMPLATE_REGISTRATION "Enables registration of TEMPLATE plugin" OFF)

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

# install

if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(template)
endif()
```

:::note
The default values of the `ENABLE_TESTS`, `ENABLE_FUNCTIONAL_TESTS` options are shared via the OpenVINO Developer Package and they are the same as for the main OpenVINO build tree. You can override them during plugin build using the command below:
:::

```bash
$ cmake -DENABLE_FUNCTIONAL_TESTS=OFF -DOpenVINODeveloperPackage_DIR=../openvino-release-build ../template-plugin
```

- `src/CMakeLists.txt` to build a plugin shared library from sources:

```cpp
set(TARGET_NAME "openvino_template_plugin")

set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/async_infer_request.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/remote_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sync_infer_request.cpp
)
set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/async_infer_request.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/attach_cache_manager_to_paged_attention.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compiled_model.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/itt.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/remote_context.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/remote_tensor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sync_infer_request.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/variable_state.hpp
)

if (NOT ENABLE_TEMPLATE_REGISTRATION)
    # Skip install and registration of template component
    set(skip_plugin SKIP_INSTALL SKIP_REGISTRATION)
endif()

# adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "TEMPLATE"
              SOURCES ${SOURCES} ${HEADERS}
              ${skip_plugin}
              VERSION_DEFINES_FOR plugin.cpp
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ov_mark_target_as_cc(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${OpenVINOTemplatePlugin_SOURCE_DIR}/include")

# link common OpenVINO Runtime libraries
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::interpreter_backend
    openvino::reference)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (ENABLE_TEMPLATE_REGISTRATION)
    # Update the plugins.xml file
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
```

:::note
`openvino::...` targets are imported from the OpenVINO Developer Package.
:::

- `tests/functional/CMakeLists.txt` to build a set of functional plugin tests:

```cpp
set(TARGET_NAME ov_template_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4305)
endif()

set(OV_TEMPLATE_TESTS_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/specific_properties_test.cpp
)

include(op_reference/sources.cmake)
include(preprocessing/sources.cmake)
include(shared_tests_instances/sources.cmake)
include(subgraph_reference/sources.cmake)
include(transformations/sources.cmake)

ov_add_test_target(
        NAME ${TARGET_NAME}
        SOURCES
            ${OV_TEMPLATE_TESTS_SRCS}
            ${OV_TEMPLATE_TESTS_OP_REFERENCE_SRCS}
            ${OV_TEMPLATE_TESTS_PREPROCESSING_SRCS}
            ${OV_TEMPLATE_TESTS_SHARED_INSTANCES_SRCS}
            ${OV_TEMPLATE_TESTS_SUBGRAPH_REFERENCE_SRCS}
            ${OV_TEMPLATE_TESTS_TRANSFORMATIONS_SRCS}
        DEPENDENCIES
            openvino_template_plugin
        LINK_LIBRARIES
            openvino::funcSharedTests
            openvino::runtime::dev
        INCLUDES
            "${OpenVINOTemplatePlugin_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/op_reference"
        DEFINES
            IN_OV_COMPONENT
        ADD_CLANG_FORMAT
        LABELS
            OV UNIT TEMPLATE
        CHECK_SOURCES_LISTED
)

find_package(OpenCV QUIET COMPONENTS core imgproc)

if(OpenCV_FOUND AND OpenCV_VERSION VERSION_GREATER_EQUAL 3.4)
    message(STATUS "Reference preprocessing: OpenCV tests are enabled")
    target_compile_definitions(${TARGET_NAME} PRIVATE OPENCV_TEMPLATE_TESTS)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_imgproc opencv_core)
else()
    message(WARNING "Reference preprocessing: OpenCV tests are disabled, because OpenCV ver. 3.4+ is not found")
endif()

if (ENABLE_INTEL_CPU)
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/executable_network/get_metric.cpp"
        PROPERTIES COMPILE_DEFINITIONS ENABLE_INTEL_CPU=1)
endif()
```

:::note
The `openvino::funcSharedTests` static library with common functional OpenVINO Plugin tests is imported via the OpenVINO Developer Package.
:::
