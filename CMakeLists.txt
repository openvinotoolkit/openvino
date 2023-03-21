# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
    # 'target_link_libraries' does not work correctly when called from
    # different directory where 'add_library' is called: CMake generates
    # incorrect OpenVINOConfig.cmake in this case
    cmake_minimum_required(VERSION 3.17)
else()
    if(CPACK_GENERATOR STREQUAL "DEB")
        # we have to use CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS variable
        cmake_minimum_required(VERSION 3.20)
    else()
        # default choice
        cmake_minimum_required(VERSION 3.13)
    endif()
endif()

project(OpenVINO DESCRIPTION "OpenVINO toolkit")

find_package(IEDevScripts REQUIRED
             PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

include(CTest)
include(cmake/features.cmake)

# These options are shared with 3rdparty plugins by means of developer package
include(cmake/dependencies.cmake)

if(ENABLE_COVERAGE)
    include(cmake/coverage.cmake)
endif()

# resolving dependencies for the project
message (STATUS "CMAKE_VERSION ......................... " ${CMAKE_VERSION})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "CMAKE_SOURCE_DIR ...................... " ${CMAKE_SOURCE_DIR})
message (STATUS "OpenVINO_SOURCE_DIR ................... " ${OpenVINO_SOURCE_DIR})
message (STATUS "OpenVINO_BINARY_DIR ................... " ${OpenVINO_BINARY_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_CXX_COMPILER_ID ................. " ${CMAKE_CXX_COMPILER_ID})
if(OV_GENERATOR_MULTI_CONFIG)
    string(REPLACE ";" " " config_types "${CMAKE_CONFIGURATION_TYPES}")
    message (STATUS "CMAKE_CONFIGURATION_TYPES ............. " ${config_types})
    unset(config_types)
    if(CMAKE_GENERATOR MATCHES "^Ninja Multi-Config$")
        message (STATUS "CMAKE_DEFAULT_BUILD_TYPE .............. " ${CMAKE_DEFAULT_BUILD_TYPE})
    endif()
else()
    message (STATUS "CMAKE_BUILD_TYPE ...................... " ${CMAKE_BUILD_TYPE})
endif()
if(CMAKE_GENERATOR_PLATFORM)
    message (STATUS "CMAKE_GENERATOR_PLATFORM .............. " ${CMAKE_GENERATOR_PLATFORM})
endif()
if(CMAKE_GENERATOR_TOOLSET)
    message (STATUS "CMAKE_GENERATOR_TOOLSET ............... " ${CMAKE_GENERATOR_TOOLSET})
endif()
if(CMAKE_TOOLCHAIN_FILE)
    message (STATUS "CMAKE_TOOLCHAIN_FILE .................. " ${CMAKE_TOOLCHAIN_FILE})
endif()
if(OV_GLIBC_VERSION)
    message (STATUS "GLIBC_VERSION ......................... " ${OV_GLIBC_VERSION})
endif()

# remove file with exported developer targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/ngraphTargets.cmake")
file(REMOVE "${CMAKE_BINARY_DIR}/InferenceEngineTargets.cmake")
file(REMOVE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")
foreach(component IN LISTS openvino_export_components)
    file(REMOVE "${CMAKE_BINARY_DIR}/${component}_dev_targets.cmake")
    file(REMOVE "${CMAKE_BINARY_DIR}/ov_${component}_dev_targets.cmake")
    unset(${component} CACHE)
endforeach()
unset(openvino_export_components CACHE)

#
# Build
#

function(openvino_developer_export_targets)
    cmake_parse_arguments(EXPORT "" "COMPONENT" "TARGETS" ${ARGN})

    if(EXPORT_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "openvino_developer_export_targets has unparsed arguments: ${EXPORT_UNPARSED_ARGUMENTS}")
    endif()

    set(${EXPORT_COMPONENT} "${${EXPORT_COMPONENT}};${EXPORT_TARGETS}")

    # to allow exporting of aliased targets with the original names
    foreach(target_name IN LISTS ${EXPORT_COMPONENT})
        if(TARGET "${target_name}")
            get_target_property(original_name ${target_name} ALIASED_TARGET)
            if(TARGET "${original_name}")
                message(STATUS "The name ${target_name} is an ALIAS for ${original_name}. "
                        "It will be exported to the OpenVINODeveloperPackage with the original name.")
                list(REMOVE_ITEM ${EXPORT_COMPONENT} ${target_name})
                list(APPEND ${EXPORT_COMPONENT} ${original_name})
            endif()
        endif()
    endforeach()

    list(REMOVE_DUPLICATES ${EXPORT_COMPONENT})
    set(${EXPORT_COMPONENT} "${${EXPORT_COMPONENT}}" CACHE INTERNAL
        "A list of OpenVINO ${EXPORT_COMPONENT} exported targets" FORCE)

    list(APPEND openvino_export_components ${EXPORT_COMPONENT})
    list(REMOVE_DUPLICATES openvino_export_components)
    set(openvino_export_components "${openvino_export_components}" CACHE INTERNAL
        "A list of OpenVINO exported components" FORCE)
endfunction()

# add target with processed tests model zoo
if (ENABLE_TESTS)
    include(cmake/test_model_zoo.cmake)
endif()

add_subdirectory(thirdparty)
add_subdirectory(src)

if(ENABLE_SAMPLES OR ENABLE_TESTS OR ENABLE_COMPILE_TOOL)
    add_subdirectory(samples)
endif()

# Enable interpreter backend for tests
if (ENABLE_TESTS OR ENABLE_TEMPLATE)
    add_subdirectory(src/plugins/template/backend)
endif()
include(cmake/extra_modules.cmake)
add_subdirectory(docs)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(licensing)

#
# CPack
#

# provides a callback function to describe each component in repo
include(cmake/packaging/packaging.cmake)

ie_cpack(${IE_CPACK_COMPONENTS_ALL})
