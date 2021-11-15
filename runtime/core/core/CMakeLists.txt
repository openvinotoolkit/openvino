# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "ov_core")

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DIN_CORE_LIBRARY)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_subdirectory(builder)
add_subdirectory(reference)
add_subdirectory(shape_inference)

# WA for Tensor implementation via ie::Blob::Ptr
set(IE_SRC_ROOT "${OpenVINO_SOURCE_DIR}/runtime/core/runtime/src")
set(IE_SHARED_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/ie_legacy/blob_factory.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/ie_legacy/ie_blob_common.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/ie_legacy/ie_layouts.cpp"
                   "${CMAKE_CURRENT_SOURCE_DIR}/src/ie_legacy/system_allocator.cpp")
set(MIXED_SRC ${IE_SHARED_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/ov_tensor.cpp")

set_property(SOURCE ${MIXED_SRC}
    APPEND PROPERTY INCLUDE_DIRECTORIES
        ${IE_SRC_ROOT}
        $<TARGET_PROPERTY:openvino::runtime,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::runtime::plugin_api,INTERFACE_INCLUDE_DIRECTORIES>)

set_source_files_properties(${MIXED_SRC}
    PROPERTIES COMPILE_DEFINITIONS IMPLEMENT_INFERENCE_ENGINE_API)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

# Create static or shared library depending on BUILD_SHARED_LIBS
add_library(${TARGET_NAME} ${LIBRARY_SRC} ${PUBLIC_HEADERS} ${IE_SHARED_SRCS})


if(COMMAND ie_faster_build)
    ie_faster_build(${TARGET_NAME}
        UNITY
        PCH PRIVATE "src/precomp.hpp"
    )
endif()

ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ${TARGET_NAME}
                           FILEDESCRIPTION "OpenVINO core library")
endif()

addVersionDefines(src/version.cpp CI_BUILD_NUMBER)

target_link_libraries(${TARGET_NAME} PRIVATE openvino::graph_builders openvino::op_references openvino::util pugixml::static ov_shape_inference)

ie_mark_target_as_cc(${TARGET_NAME})

ov_ncc_naming_style(FOR_TARGET ${TARGET_NAME}
                    INCLUDE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # ov_core is linked against graph_builders, op_references, ov_shape_inference static libraries
    # which include ov_core headers with dllimport attribute. Linker complains about it
    # but no way to fix this: linking with no attribute defaults to dllexport and we have
    # multiple defitions for ov_core symbols.
    #
    # The possible way is to use object libraries for graph_builders, op_references
    # but it's not convinient since these libraries are exported from build tree
    # and it's better to use them as static libraries in 3rd party projects
    if(BUILD_SHARED_LIBS)
        set(link_type PRIVATE)
    else()
        set(link_type PUBLIC)
    endif()

    target_link_options(${TARGET_NAME} ${link_type} "/IGNORE:4217,4286")
endif()

# some sources are located in ov_core, while headers are in openvino::transformations
file(GLOB_RECURSE smart_reshape_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/smart_reshape/*.cpp)
file(GLOB_RECURSE rt_info_srcs ${CMAKE_CURRENT_SOURCE_DIR}/src/pass/rt_info/*.cpp)
set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/pass/convert_precision.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pass/convert_fp32_to_fp16.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pass/init_node_info.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/pass/serialize.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/op/type_relaxed.cpp"
                            "${CMAKE_CURRENT_SOURCE_DIR}/src/function.cpp" # for SmartReshape
                            ${smart_reshape_srcs} ${rt_info_srcs}
        PROPERTIES INCLUDE_DIRECTORIES $<TARGET_PROPERTY:openvino::transformations,INTERFACE_INCLUDE_DIRECTORIES>)

# Defines macro in C++ to load backend plugin
target_include_directories(${TARGET_NAME} PUBLIC $<BUILD_INTERFACE:${OV_CORE_INCLUDE_PATH}>
                                  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(ngraph::ngraph ALIAS ${TARGET_NAME})
add_library(openvino::core ALIAS ${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE ${CMAKE_DL_LIBS})

#-----------------------------------------------------------------------------------------------
# Export for build tree
#-----------------------------------------------------------------------------------------------

set_target_properties(${TARGET_NAME} PROPERTIES EXPORT_NAME core)
export(TARGETS ${TARGET_NAME} NAMESPACE openvino::
       APPEND FILE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

install(TARGETS ${TARGET_NAME} EXPORT OpenVINOTargets
        RUNTIME DESTINATION ${IE_CPACK_RUNTIME_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_ARCHIVE_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core
        INCLUDES DESTINATION runtime/include)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION "runtime/include"
        COMPONENT core_dev
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h")

configure_package_config_file(${OpenVINO_SOURCE_DIR}/cmake/templates/ngraphConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfig.cmake
    INSTALL_DESTINATION cmake)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ngraphConfigVersion.cmake
    VERSION ${IE_VERSION_MAJOR}.${IE_VERSION_MINOR}.${IE_VERSION_PATCH}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfigVersion.cmake
        DESTINATION "runtime/cmake"
        COMPONENT core_dev)
