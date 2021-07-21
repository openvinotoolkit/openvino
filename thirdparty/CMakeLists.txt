# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

add_subdirectory(ittapi EXCLUDE_FROM_ALL)
add_subdirectory(itt_collector EXCLUDE_FROM_ALL)
add_subdirectory(zlib EXCLUDE_FROM_ALL)
add_subdirectory(cnpy EXCLUDE_FROM_ALL)
add_subdirectory(ocl EXCLUDE_FROM_ALL)

add_subdirectory(xbyak EXCLUDE_FROM_ALL)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS xbyak)

#
# Pugixml
#

if(ENABLE_SYSTEM_PUGIXML)
    find_package(PugiXML REQUIRED)
    set_property(TARGET pugixml PROPERTY IMPORTED_GLOBAL TRUE)
    add_library(pugixml::static ALIAS pugixml)
else()
    function(ie_build_pugixml)
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory(pugixml EXCLUDE_FROM_ALL)
        set_property(TARGET pugixml-static PROPERTY EXPORT_NAME pugixml)
        openvino_developer_export_targets(COMPONENT openvino_common TARGETS pugixml::static)
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
set_target_properties(ade fluid PROPERTIES FOLDER thirdparty)
openvino_developer_export_targets(COMPONENT openvino_common TARGETS ade fluid)

#
# Gflags
#

function(add_gflags)
    # common gflags settings
    set(GFLAGS_IS_SUBPROJECT TRUE)
    set(HAVE_SYS_STAT_H 1)
    set(HAVE_INTTYPES_H 1)
    set(INTTYPES_FORMAT C99)
    set(BUILD_TESTING OFF)
    set(BUILD_SHARED_LIBS OFF)

    add_subdirectory(gflags EXCLUDE_FROM_ALL)
    set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
    openvino_developer_export_targets(COMPONENT openvino_common TARGETS gflags)
endfunction()

add_gflags()

#
# Google Tests framework
#

function(add_gtest_libraries)
    set(gtest_force_shared_crt ON CACHE BOOL "disable static CRT for google test")
    if (UNIX)
        ie_add_compiler_flags(-Wno-undef)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            ie_add_compiler_flags(-Wno-deprecated-copy)
        endif()
    endif()

    set(BUILD_SHARED_LIBS OFF)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    add_subdirectory(gtest EXCLUDE_FROM_ALL)

    get_target_property(gtest_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gtest_include_dirs}")

    get_target_property(gmock_include_dirs gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${gmock_include_dirs};${gmock_SOURCE_DIR}/include")

    # If we have specified /Z7 option, remove -Zi option which comes from gtest
    if (WIN32)
        set(gtest_targets "gtest;gtest_main;gmock;gmock_main")
        foreach(target_name ${gtest_targets})
            if(TARGET "${target_name}")
                get_target_property(_target_cxx_flags ${target_name} COMPILE_OPTIONS)
                if(_target_cxx_flags)
                    if(CMAKE_CXX_FLAGS_DEBUG MATCHES ".+/Z7.+" OR CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES ".+/Z7.+")
                        string(REPLACE "-Zi" " " _target_cxx_flags ${_target_cxx_flags})
                        message(STATUS "Removing -Zi flag from target " ${target_name})
                        set_target_properties(${target_name} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
                    endif()
                endif()
            endif()
        endforeach()
    endif()
    set_target_properties(gtest gtest_main gmock gmock_main
                          PROPERTIES FOLDER thirdparty)
    
    openvino_developer_export_targets(COMPONENT inference_engine_tests
                                      TARGETS gmock gmock_main gtest gtest_main)
endfunction()

if(NGRAPH_UNIT_TEST_ENABLE OR ENABLE_TESTS)
    add_gtest_libraries()
endif()

#
# Protobuf
#

if(NGRAPH_PDPD_FRONTEND_ENABLE OR NGRAPH_ONNX_IMPORT_ENABLE)
    if(NGRAPH_USE_SYSTEM_PROTOBUF)
        set(Protobuf_USE_STATIC_LIBS ON)
        if(VERBOSE_BUILD)
            set(Protobuf_DEBUG ON)
        endif()
        find_package(Protobuf REQUIRED)
        if(NGRAPH_USE_PROTOBUF_LITE)
            set(Protobuf_LIBRARIES protobuf::libprotobuf-lite)
        else()
            set(Protobuf_LIBRARIES protobuf::libprotobuf)
        endif()
        set(SYSTEM_PROTOC protobuf::protoc)
        set(PROTOC_EXECUTABLE ${SYSTEM_PROTOC})

        foreach(target ${SYSTEM_PROTOC} ${Protobuf_LIBRARIES})
            set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
        endforeach()
    else()
        add_subdirectory(protobuf)
    endif()

    # forward variables used in the other places
    set(SYSTEM_PROTOC ${SYSTEM_PROTOC} PARENT_SCOPE)
    set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
    set(Protobuf_LIBRARIES ${Protobuf_LIBRARIES} PARENT_SCOPE)
    set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIRS} PARENT_SCOPE)

    # set public / interface compile options
    foreach(target IN LISTS Protobuf_LIBRARIES)
        set(link_type PUBLIC)
        if(NGRAPH_USE_SYSTEM_PROTOBUF)
            set(link_type INTERFACE)
        endif()
        if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
            target_compile_options(${target} ${link_type} -Wno-undef)
        endif()
    endforeach()

    # version checks
    if(protobuf_VERSION VERSION_LESS "3.9" AND NGRAPH_USE_PROTOBUF_LITE)
        message(FATAL_ERROR "Minimum supported version of protobuf-lite library is 3.9.0 (provided ${protobuf_VERSION})")
    endif()
endif()

#
# ONNX
#

if(NGRAPH_ONNX_IMPORT_ENABLE)
    add_subdirectory(onnx)
endif()

#
# Install
#

ie_cpack_add_component(cpp_samples_deps)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/gflags
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/zlib
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cnpy
        DESTINATION ${IE_CPACK_IE_DIR}/samples/cpp/thirdparty
        COMPONENT cpp_samples_deps
        USE_SOURCE_PERMISSIONS)
