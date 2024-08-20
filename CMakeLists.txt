# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
    # 3.17: 'target_link_libraries' does not work correctly when called from
    # different directory where 'add_library' is called: CMake generates
    # incorrect OpenVINOConfig.cmake in this case
    # 3.18: add_library cannot create ALIAS for non-GLOBAL targets
    cmake_minimum_required(VERSION 3.18)
else()
    if(CPACK_GENERATOR STREQUAL "DEB")
        # we have to use CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS variable
        cmake_minimum_required(VERSION 3.20)
    else()
        if(WIN32)
            # 3.16: FindPython3.cmake can find Python via -DPython3_EXECUTABLE
            # 3.18: FindPython3.cmake can find Python automatically from virtualenv
            cmake_minimum_required(VERSION 3.16)
        else()
            # 3.13: default choice
            cmake_minimum_required(VERSION 3.13)
        endif()
    endif()
endif()

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW) # Enables use of MSVC_RUNTIME_LIBRARY
endif()

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0149)
    # VS generator looks for most recent Windows SDK, ignoring
    # CMAKE_SYSTEM_VERSION and allowing override by WindowsSDKVersion
    # environment variable. New in 3.27. This is to allow override
    # in the Windows CI builds.
    cmake_policy(SET CMP0149 NEW)
endif()

project(OpenVINO
        DESCRIPTION "OpenVINO toolkit"
        HOMEPAGE_URL "https://docs.openvino.ai/2024/home.html"
        LANGUAGES C CXX)

find_package(OpenVINODeveloperScripts REQUIRED
             PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

include(cmake/features.cmake)

# These options are shared with 3rdparty plugins by means of developer package
include(cmake/dependencies.cmake)

if(ENABLE_COVERAGE)
    include(cmake/coverage.cmake)
endif()

if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET AND
    CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 10.15)
    message(FATAL_ERROR "OpenVINO requires MACOSX_DEPLOYMENT_TARGET at least 10.15, specified ${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()

# resolving dependencies for the project
message (STATUS "CMAKE_VERSION ......................... " ${CMAKE_VERSION})
message (STATUS "CMAKE_CROSSCOMPILING .................. " ${CMAKE_CROSSCOMPILING})
message (STATUS "OpenVINO_SOURCE_DIR ................... " ${OpenVINO_SOURCE_DIR})
message (STATUS "OpenVINO_BINARY_DIR ................... " ${OpenVINO_BINARY_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CPACK_GENERATOR ....................... " ${CPACK_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})
message (STATUS "CMAKE_CXX_COMPILER_ID ................. " ${CMAKE_CXX_COMPILER_ID})
message (STATUS "CMAKE_CXX_STANDARD .................... " ${CMAKE_CXX_STANDARD})
if(OV_GENERATOR_MULTI_CONFIG)
    string(REPLACE ";" " " config_types "${CMAKE_CONFIGURATION_TYPES}")
    message (STATUS "CMAKE_CONFIGURATION_TYPES ............. " ${config_types})
    unset(config_types)
    if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
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
if(NOT OV_LIBC_VERSION VERSION_EQUAL 0.0)
    message (STATUS "LIBC_VERSION .......................... " ${OV_LIBC_VERSION})
endif()
if(DEFINED OPENVINO_STDLIB)
    message (STATUS "STDLIB ................................ " ${OPENVINO_STDLIB})
endif()

# remove file with exported targets to force its regeneration
file(REMOVE "${CMAKE_BINARY_DIR}/OpenVINOTargets.cmake")

# remove exported developer targets files to force its regeneration
macro(ov_clean_developer_package_targets)
    file(REMOVE "${CMAKE_BINARY_DIR}/openvino_developer_package_targets.cmake")
    unset(_OPENVINO_DEVELOPER_PACKAGE_TARGETS CACHE)
    unset(openvino_installed_targets CACHE)
endmacro()
ov_clean_developer_package_targets()

function(ov_developer_package_export_targets)
    cmake_parse_arguments(EXPORT "" "TARGET;INSTALL_DESTIONATION" "INSTALL_INCLUDE_DIRECTORIES" ${ARGN})

    # to allow exporting of aliased targets with the original names
    if(TARGET "${EXPORT_TARGET}")
        get_target_property(original_name ${EXPORT_TARGET} ALIASED_TARGET)
        if(TARGET "${original_name}")
            # replace target with its original name
            set(EXPORT_TARGET ${original_name})
        endif()
        list(APPEND _OPENVINO_DEVELOPER_PACKAGE_TARGETS ${EXPORT_TARGET})

        if(EXPORT_INSTALL_INCLUDE_DIRECTORIES)
            if(NOT EXPORT_INSTALL_DESTIONATION)
                set(EXPORT_INSTALL_DESTIONATION "developer_package/include/${EXPORT_TARGET}")
            endif()

            target_include_directories(${EXPORT_TARGET} INTERFACE "$<INSTALL_INTERFACE:${EXPORT_INSTALL_DESTIONATION}>")

            foreach(install_dir IN LISTS EXPORT_INSTALL_INCLUDE_DIRECTORIES)
                install(DIRECTORY "${install_dir}"
                        DESTINATION "${EXPORT_INSTALL_DESTIONATION}"
                        COMPONENT developer_package EXCLUDE_FROM_ALL)
            endforeach()
        endif()
    else()
        message(FATAL_ERROR "Internal error: ${target_name} does not represent a cmake target")
    endif()

    list(REMOVE_DUPLICATES _OPENVINO_DEVELOPER_PACKAGE_TARGETS)
    set(_OPENVINO_DEVELOPER_PACKAGE_TARGETS "${_OPENVINO_DEVELOPER_PACKAGE_TARGETS}" CACHE INTERNAL
        "A list of OpenVINO Developer Package exported targets" FORCE)
endfunction()

#
# Build
#

if(ENABLE_TESTS)
    # add target with processed tests model zoo
    include(cmake/test_model_zoo.cmake)
endif()

include(thirdparty/dependencies.cmake)
add_subdirectory(src)

if(ENABLE_SAMPLES OR ENABLE_TESTS)
    add_subdirectory(samples)
endif()

# Enable interpreter backend for tests
if(ENABLE_TESTS OR ENABLE_TEMPLATE)
    add_subdirectory(src/plugins/template/backend)
endif()

include(cmake/extra_modules.cmake)
add_subdirectory(docs)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(licensing)

if(ENABLE_TESTS)
    # layers and other more high-level / e2e tests
    add_subdirectory(tests)
endif()

#
# CPack
#

# provides a callback function to describe each component in repo
include(cmake/packaging/packaging.cmake)

ov_cpack(${OV_CPACK_COMPONENTS_ALL})