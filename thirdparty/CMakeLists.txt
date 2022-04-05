# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_subdirectory(ittapi)
add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)
if(ENABLE_INTEL_GPU)
    add_subdirectory(ocl)
endif()

add_subdirectory(xbyak EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)
ov_install_static_lib(xbyak openvino_common)

#
# zlib
#

if(NOT CMAKE_CROSSCOMPILING)
    find_host_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_search_module(zlib QUIET
                        IMPORTED_TARGET GLOBAL
                        zlib)
        if(zlib_FOUND)
            add_library(zlib::zlib ALIAS PkgConfig::zlib)
            message(STATUS "zlib (${zlib_VERSION}) is found at ${zlib_PREFIX}")
        endif()
    endif()
endif()

if(NOT zlib_FOUND)
    add_subdirectory(zlib EXCLUDE_FROM_ALL)
endif()

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    find_package(PugiXML REQUIRED)
    set_target_properties(pugixml PROPERTIES IMPORTED_GLOBAL ON)
    add_library(pugixml::static ALIAS pugixml)
else()
    function(ie_build_pugixml)
        function(ie_build_pugixml_static)
            set(BUILD_SHARED_LIBS OFF)
            add_subdirectory(pugixml EXCLUDE_FROM_ALL)
        endfunction()
        ie_build_pugixml_static()
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml::static)
        ov_install_static_lib(pugixml-static openvino_common)
    endfunction()

    ie_build_pugixml()
endif()

#
# Fluid, G-API, OpenCV HAL
#

add_library(ocv_hal INTERFACE)
target_include_directories(ocv_hal INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/ocv")

add_subdirectory(ade EXCLUDE_FROM_ALL)
add_subdirectory(fluid/modules/gapi EXCLUDE_FROM_ALL)

if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11 AND NOT OV_COMPILER_IS_CLANG)
    target_compile_options(fluid PRIVATE "-Wno-maybe-uninitialized")
endif()

set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade fluid)

ov_install_static_lib(ade openvino_common)
ov_install_static_lib(fluid openvino_common)

#
# Gflags
#

find_package(gflags QUIET COMPONENTS nothreads_shared)
if(gflags_FOUND)
    set_target_properties(gflags PROPERTIES IMPORTED_GLOBAL ON)
    message(STATUS "gflags (${gflags_VERSION}) is found at ${gflags_DIR}")
else()
    add_subdirectory(gflags EXCLUDE_FROM_ALL)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags)
endif()

#
# Google Tests framework
#

if(ENABLE_OV_CORE_UNIT_TESTS OR ENABLE_TESTS)
    add_subdirectory(gtest EXCLUDE_FROM_ALL)

    openvino_developer_export_targets(COMPONENT tests
                                      TARGETS gmock gmock_main gtest gtest_main)
endif()

#
# Protobuf
#

if(ENABLE_OV_PADDLE_FRONTEND OR ENABLE_OV_ONNX_FRONTEND OR ENABLE_OV_TF_FRONTEND)
    if(ENABLE_SYSTEM_PROTOBUF)
        set(Protobuf_USE_STATIC_LIBS ON)
        if(VERBOSE_BUILD)
            set(Protobuf_DEBUG ON)
        endif()
        find_package(Protobuf 3.18.2 REQUIRED)
        set(Protobuf_LITE_LIBRARIES protobuf::libprotobuf-lite)
        set(Protobuf_LIBRARIES protobuf::libprotobuf)
        set(SYSTEM_PROTOC protobuf::protoc)
        set(PROTOC_EXECUTABLE ${SYSTEM_PROTOC})

        foreach(target ${SYSTEM_PROTOC} ${Protobuf_LIBRARIES} ${Protobuf_LITE_LIBRARIES})
            set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
        endforeach()
    else()
        add_subdirectory(protobuf EXCLUDE_FROM_ALL)
    endif()

    # forward variables used in the other places
    set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
    set(Protobuf_LIBRARIES ${Protobuf_LIBRARIES} PARENT_SCOPE)
    set(Protobuf_LITE_LIBRARIES ${Protobuf_LITE_LIBRARIES} PARENT_SCOPE)
    set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)
    set(Protobuf_IN_FRONTEND ON PARENT_SCOPE)

    # set public / interface compile options
    foreach(target IN LISTS Protobuf_LITE_LIBRARIES Protobuf_LIBRARIES)
        set(link_type PUBLIC)
        if(ENABLE_SYSTEM_PROTOBUF)
            set(link_type INTERFACE)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
            target_compile_options(${target} ${link_type} -Wno-undef)
        endif()
    endforeach()
endif()

#
# ONNX
#

if(ENABLE_OV_ONNX_FRONTEND)
    add_subdirectory(onnx)
endif()

#
# nlohmann json
#

add_subdirectory(json)

openvino_developer_export_targets(COMPONENT openvino_common
                                  TARGETS nlohmann_json nlohmann_json_schema_validator)

ov_install_static_lib(nlohmann_json openvino_common)
ov_install_static_lib(nlohmann_json_schema_validator openvino_common)

#
# Install
#

if(CPACK_GENERATOR STREQUAL "DEB")
    # These libraries are dependencies for openvino-samples-2022.1 package

    if(NOT gflags_FOUND)
        message(FATAL_ERROR "gflags must be used as a debian package. Install libflags-dev")
    endif()
    if(NOT zlib_FOUND)
        message(FATAL_ERROR "zlib must be used as a debian package. Install zlib1g-dev")
    endif()
else()
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gflags
            DESTINATION "samples/cpp/thirdparty"
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zlib
            DESTINATION "samples/cpp/thirdparty"
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/json
            DESTINATION "samples/cpp/thirdparty"
            COMPONENT cpp_samples
            USE_SOURCE_PERMISSIONS)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION "samples/cpp/thirdparty"
        COMPONENT cpp_samples
        USE_SOURCE_PERMISSIONS)
