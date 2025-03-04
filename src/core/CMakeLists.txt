# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# enable LTO globally for all libraries below
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

add_definitions(-DIN_OV_CORE_LIBRARY)

set(OV_CORE_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OV_CORE_DEV_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dev_api)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${OV_CORE_INCLUDE_PATH}/*.hpp)
file(GLOB_RECURSE DEV_HEADERS ${OV_CORE_DEV_API_PATH}/*.hpp)

add_subdirectory(reference)
add_subdirectory(shape_inference)

set(MIXED_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/itensor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/tensor.cpp")

set_property(SOURCE ${MIXED_SRC}
    APPEND PROPERTY INCLUDE_DIRECTORIES
        $<TARGET_PROPERTY:openvino_runtime_obj,SOURCE_DIR>/src
        $<TARGET_PROPERTY:openvino_runtime_obj,SOURCE_DIR>/dev_api
        $<TARGET_PROPERTY:openvino_runtime_obj,SOURCE_DIR>/include)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

#
# Create openvino_core_dev library
#

add_library(openvino_core_dev INTERFACE)
add_library(openvino::core::dev ALIAS openvino_core_dev)

target_include_directories(openvino_core_dev INTERFACE
    $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/core/dev_api>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/frontends/common/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/common/transformations/include>
    $<BUILD_INTERFACE:${OpenVINO_SOURCE_DIR}/src/common/low_precision_transformations/include>)

target_include_directories(openvino_core_dev SYSTEM INTERFACE
    $<BUILD_INTERFACE:$<$<TARGET_EXISTS:xbyak::xbyak>:$<TARGET_PROPERTY:xbyak::xbyak,INTERFACE_INCLUDE_DIRECTORIES>>>)

target_link_libraries(openvino_core_dev INTERFACE openvino::itt openvino::util)

set_target_properties(openvino_core_dev PROPERTIES EXPORT_NAME core::dev)
ov_developer_package_export_targets(TARGET openvino::core::dev
                                    INSTALL_INCLUDE_DIRECTORIES
                                        "${OV_CORE_DEV_API_PATH}/"
                                        "${OpenVINO_SOURCE_DIR}/src/common/transformations/include/"
                                        "${OpenVINO_SOURCE_DIR}/src/common/low_precision_transformations/include/")

# Install interface libraries for case BUILD_SHARED_LIBS=OFF
ov_install_static_lib(openvino_core_dev ${OV_CPACK_COMP_CORE})

#
# Create static or shared library depending on BUILD_SHARED_LIBS
#

add_library(openvino_core_obj OBJECT ${LIBRARY_SRC} ${PUBLIC_HEADERS})

if(ENABLE_SYSTEM_PUGIXML)
    # system pugixml has /usr/include as include directories
    # we cannot use them as system ones, leads to compilation errors
    set_target_properties(openvino_core_obj PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
endif()

target_compile_definitions(openvino_core_obj PRIVATE IMPLEMENT_OPENVINO_API XBYAK_NO_OP_NAMES XBYAK64)

ov_build_target_faster(openvino_core_obj
    UNITY
    PCH PRIVATE "src/precomp.hpp")

ov_add_version_defines(src/version.cpp openvino_core_obj)

ov_set_threading_interface_for(openvino_core_obj)

target_link_libraries(openvino_core_obj PRIVATE openvino::reference openvino::util
                                         openvino::pugixml openvino::shape_inference openvino::core::dev)

ov_mark_target_as_cc(openvino_core_obj)

# openvino_core is public API => need to mark this library as important for ABI free
ov_abi_free_target(openvino_core_obj)

ov_ncc_naming_style(FOR_TARGET openvino_core_obj
                    SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")

ov_add_clang_format_target(openvino_core_clang FOR_SOURCES ${LIBRARY_SRC} ${PUBLIC_HEADERS} ${DEV_HEADERS})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(openvino_core_obj PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # openvino_core is linked against openvino::reference, openvino::shape_inference static libraries
    # which include openvino_core headers with dllimport attribute. Linker complains about it
    # but no way to fix this: linking with no attribute defaults to dllexport and we have
    # multiple defitions for openvino_core symbols.
    #
    # The possible way is to use object libraries for openvino::reference
    # but it's not convinient since these libraries are exported from build tree
    # and it's better to use them as static libraries in 3rd party projects
    if(BUILD_SHARED_LIBS)
        set(link_type PRIVATE)
    else()
        set(link_type PUBLIC)
    endif()

    target_link_options(openvino_core_obj ${link_type} "/IGNORE:4217,4286")
endif()

# TODO: try to remove this and move smart reshape to transformations
# some sources are located in openvino_core, while headers are in openvino_transformations
file(GLOB_RECURSE smart_reshape_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/smart_reshape/*.cpp)
file(GLOB_RECURSE rt_info_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/rt_info/*.cpp)
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/pass/convert_precision.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pass/convert_fp32_to_fp16.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pass/init_node_info.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pass/serialize.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/op/type_relaxed.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/preprocess/preprocess_steps_impl.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp" # for SmartReshape
                            ${smart_reshape_srcs} ${rt_info_srcs}
        PROPERTIES INCLUDE_DIRECTORIES $<TARGET_PROPERTY:openvino::core::dev,INTERFACE_INCLUDE_DIRECTORIES>)

# Defines macro in C++ to load backend plugin
target_include_directories(openvino_core_obj PUBLIC $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>
                                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${OV_CPACK_INCLUDEDIR}
        COMPONENT ${OV_CPACK_COMP_CORE_DEV}
        ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL}
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h")
