# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.10)

# Enable CMAKE_<LANG>_COMPILER_ID AppleClang
if(POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif()

project(Samples)

get_property(OV_GENERATOR_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(CMAKE_GENERATOR MATCHES "^Ninja Multi-Config$")
    # Ninja-Multi specific, see:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_DEFAULT_BUILD_TYPE.html
    set(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "CMake default build type")
elseif(NOT OV_GENERATOR_MULTI_CONFIG)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release;Debug;RelWithDebInfo;MinSizeRel")
endif()

# disable FindPkgConfig.cmake for Android
if(ANDROID)
    # Android toolchain does not provide pkg-config file. So, cmake mistakenly uses
    # build system pkg-config executable, which finds packages on build system. Such
    # libraries cannot be linked into Android binaries.
    set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig ON)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT BIN_FOLDER)
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64" # Windows detects Intel's 64-bit CPU as AMD64
        OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})

    if(UNIX)
        set(BIN_FOLDER "${BIN_FOLDER}/${CMAKE_BUILD_TYPE}")
    endif()
endif()

if(OpenVINO_SOURCE_DIR)
    # in case if samples are built from IE repo
    set(IE_MAIN_SAMPLES_DIR "${OpenVINO_SOURCE_DIR}")
    set(OpenVINO_DIR "${CMAKE_BINARY_DIR}")
else()
    # in case if samples are built out of IE repo
    set(IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") # no asynchronous structured exception handling
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    # disable some noisy warnings
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267 /wd4819")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    if(WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:177")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable:177")
    endif()
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
    set(AARCH64 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(ARM ON)
endif()

if(ARM AND NOT (CMAKE_CROSSCOMPILING OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    add_compile_options(-march=armv7-a+fp)
endif()

####################################
## to use C++11; can overwritten via cmake command line
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_EXTENSIONS OFF)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
####################################

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common/utils")
    # gflags is used only in C++ samples
    set(gflags_required ON)
endif()

if(TARGET gflags)
    set(GFLAGS_TARGET gflags)
elseif(gflags_required)
    if(EXISTS /etc/debian_version)
        set(gflags_component nothreads_static)
    else()
        set(gflags_component shared)
    endif()
    find_package(gflags QUIET OPTIONAL_COMPONENTS ${gflags_component})
    if(gflags_FOUND)
        if(TARGET ${GFLAGS_TARGET})
            # nothing
        elseif(TARGET gflags_nothreads-static)
            # Debian 9: gflag_component is ignored
            set(GFLAGS_TARGET gflags_nothreads-static)
        elseif(TARGET gflags-shared)
            # gflags shared case for CentOS / RHEL / Fedora
            set(GFLAGS_TARGET gflags-shared)
        elseif(TARGET gflags::gflags)
            # gflags case for Conan
            set(GFLAGS_TARGET gflags::gflags)
        else()
            message(FATAL_ERROR "Internal error: failed to find imported target 'gflags' using '${gflags_component}' component")
        endif()

        message(STATUS "gflags (${gflags_VERSION}) is found at ${gflags_DIR} using '${gflags_component}' component")
    endif()

    if(NOT gflags_FOUND)
        set(GFLAGS_TARGET gflags_nothreads_static)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags")
            add_subdirectory(thirdparty/gflags EXCLUDE_FROM_ALL)
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/gflags")
            # Allow running samples CMakeLists.txt as stand alone from openvino sources
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/gflags"
                             "${CMAKE_CURRENT_BINARY_DIR}/thirdparty/gflags" EXCLUDE_FROM_ALL)
        else()
            message(FATAL_ERROR "Failed to find 'gflags' library using '${gflags_component}' component")
        endif()
    endif()
endif()

# include common utils
add_subdirectory(common)

# samples build can be switched off during whole OpenVINO build
if (DEFINED OpenVINO_SOURCE_DIR AND NOT ENABLE_SAMPLES)
    return()
endif()

function(add_samples_to_build)
    # check each passed sample subdirectory
    foreach (dir ${ARGN})
        if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
            # check if a subdirectory contains CMakeLists.txt. In this case we can build it.
            file(GLOB is_sample_dir "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
            if(is_sample_dir)
                # check if specified sample/demo is found.
                if (BUILD_SAMPLE_NAME)
                    list(FIND BUILD_SAMPLE_NAME ${dir} index)
                endif()
                if (index EQUAL -1)
                    message(STATUS "${dir} SKIPPED")
                else()
                    # Include subdirectory to the project.
                    add_subdirectory(${dir})
                endif()
            endif()
        endif()
    endforeach()
endfunction(add_samples_to_build)

include(CMakeParseArguments)

#
# ie_add_sample(NAME <target name>
#               SOURCES <source files>
#               [HEADERS <header files>]
#               [INCLUDE_DIRECTORIES <include dir>]
#               [DEPENDENCIES <dependencies>]
#               [EXCLUDE_CLANG_FORMAT]
#
macro(ie_add_sample)
    set(options EXCLUDE_CLANG_FORMAT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_SAMPLE_SOURCES})
    if(IE_SAMPLE_HEADERS)
        source_group("include" FILES ${IE_SAMPLE_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_SAMPLE_NAME} ${IE_SAMPLE_SOURCES} ${IE_SAMPLE_HEADERS})

    set(folder_name cpp_samples)
    if(IE_SAMPLE_NAME MATCHES ".*_c$")
        set(c_sample ON)
        set(folder_name c_samples)
    endif()

    # for cross-compilation with gflags
    find_package(Threads REQUIRED)

    find_package(OpenVINO REQUIRED COMPONENTS Runtime)

    # Conan does not generate openvino::runtime::c target
    if(c_sample AND TARGET openvino::runtime::c)
        set(ov_link_libraries openvino::runtime::c)
    else()
        set(ov_link_libraries openvino::runtime)
    endif()

    set_target_properties(${IE_SAMPLE_NAME} PROPERTIES FOLDER ${folder_name}
                                                       COMPILE_PDB_NAME ${IE_SAMPLE_NAME})

    if(IE_SAMPLE_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_SAMPLE_NAME} PRIVATE ${IE_SAMPLE_INCLUDE_DIRECTORIES})
    endif()
    target_include_directories(${IE_SAMPLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${ov_link_libraries} Threads::Threads ${IE_SAMPLE_DEPENDENCIES})

    install(TARGETS ${IE_SAMPLE_NAME}
            RUNTIME DESTINATION samples_bin/
            COMPONENT samples_bin
            EXCLUDE_FROM_ALL)

    # create global target with all samples / demo apps
    if(NOT TARGET ie_samples)
        add_custom_target(ie_samples ALL)
    endif()
    add_dependencies(ie_samples ${IE_SAMPLE_NAME})

    if(COMMAND ov_add_clang_format_target AND NOT IE_SAMPLE_EXCLUDE_CLANG_FORMAT)
        ov_add_clang_format_target(${IE_SAMPLE_NAME}_clang FOR_SOURCES ${IE_SAMPLE_SOURCES} ${IE_SAMPLE_HEADERS})
    endif()
    if(COMMAND ov_ncc_naming_style AND NOT c_sample)
        ov_ncc_naming_style(FOR_TARGET "${IE_SAMPLE_NAME}"
                            SOURCE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endmacro()

# collect all samples subdirectories
file(GLOB samples_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# skip building of unnecessary subdirectories
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
    list(REMOVE_ITEM samples_dirs thirdparty)
endif()
list(REMOVE_ITEM samples_dirs common)

add_samples_to_build(${samples_dirs})
