# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED OpenVINODeveloperScripts_DIR )
    message(FATAL_ERROR "OpenVINODeveloperScripts_DIR  is not defined")
endif()

set(IEDevScripts_DIR "${OpenVINODeveloperScripts_DIR}") # for BW compatibility

# disable FindPkgConfig.cmake for Android
if(ANDROID)
    # Android toolchain does not provide pkg-config file. So, cmake mistakenly uses
    # build system pkg-config executable, which finds packages on build system. Such
    # libraries cannot be linked into Android binaries.
    set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig ON)
endif()

macro(ov_set_if_not_defined var value)
    if(NOT DEFINED ${var})
        set(${var} ${value})
    endif()
endmacro()

set(OLD_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${OpenVINODeveloperScripts_DIR}")

function(ov_set_ci_build_number)
    include(version)
    ov_parse_ci_build_number("${CMAKE_SOURCE_DIR}")

    foreach(var CI_BUILD_NUMBER OpenVINO_VERSION OpenVINO_SOVERSION OpenVINO_VERSION_SUFFIX
                OpenVINO_VERSION_MAJOR OpenVINO_VERSION_MINOR OpenVINO_VERSION_PATCH OpenVINO_VERSION_BUILD)
        if(NOT DEFINED ${var})
            message(FATAL_ERROR "${var} version component is not defined")
        endif()
        set(${var} "${${var}}" PARENT_SCOPE)
    endforeach()
endfunction()

# explicitly configure FindPython3.cmake to find python3 in virtual environment first
ov_set_if_not_defined(Python3_FIND_STRATEGY LOCATION)

include(features)

ov_set_ci_build_number()

#
# Detect target
#

include(target_flags)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_FOLDER)
if(X86_64)
    set(ARCH_FOLDER intel64)
elseif(X86)
    set(ARCH_FOLDER ia32)
elseif(MSVC AND ARM)
    set(ARCH_FOLDER arm)
elseif((MSVC OR APPLE) AND AARCH64)
    set(ARCH_FOLDER arm64)
elseif(UNIVERSAL2)
    set(ARCH_FOLDER universal2)
endif()

#
# Prepare temporary folder
#

function(ov_set_temp_directory temp_variable source_tree_dir)
    if(DEFINED OV_TEMP)
        message(STATUS "OV_TEMP cmake variable is set : ${OV_TEMP}")
        file(TO_CMAKE_PATH ${OV_TEMP} temp)
    elseif (DEFINED ENV{DL_SDK_TEMP} AND NOT $ENV{DL_SDK_TEMP} STREQUAL "")
        message(STATUS "DL_SDK_TEMP environment is set : $ENV{DL_SDK_TEMP}")
        file(TO_CMAKE_PATH $ENV{DL_SDK_TEMP} temp)
    else ()
        set(temp ${source_tree_dir}/temp)
    endif()

    set("${temp_variable}" "${temp}" CACHE PATH "Path to temp directory")
    if(ALTERNATIVE_PATH)
        set(ALTERNATIVE_PATH "${ALTERNATIVE_PATH}" PARENT_SCOPE)
    endif()
endfunction()

#
# For cross-compilation
#

include(cross_compile/find_commands)
include(cross_compile/python_helpers)
include(cross_compile/native_compile)

#
# Common scripts
#

include(coverage/coverage)
include(shellcheck/shellcheck)

# printing debug messages
include(debug)

if(OS_FOLDER)
    message ("**** OS FOLDER IS: [${OS_FOLDER}]")
    if(OS_FOLDER STREQUAL "ON")
        message ("**** USING OS FOLDER: [${CMAKE_SYSTEM_NAME}]")
        set(BIN_FOLDER "bin/${CMAKE_SYSTEM_NAME}/${ARCH_FOLDER}")
    else()
        set(BIN_FOLDER "bin/${OS_FOLDER}/${ARCH_FOLDER}")
    endif()
else()
    set(BIN_FOLDER "bin/${ARCH_FOLDER}")
endif()

if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
    # 'Ninja Multi-Config' specific, see:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_DEFAULT_BUILD_TYPE.html
    set(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "CMake default build type")
elseif(NOT OV_GENERATOR_MULTI_CONFIG)
    if(NOT CMAKE_BUILD_TYPE)
        # default value
        set(CMAKE_BUILD_TYPE "Release")
    endif()
    set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "CMake build type")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release;Debug;RelWithDebInfo;MinSizeRel")
endif()

if(USE_BUILD_TYPE_SUBFOLDER)
    set(BIN_FOLDER "${BIN_FOLDER}/${CMAKE_BUILD_TYPE}")
endif()

# allow to override default OUTPUT_ROOT root
if(NOT DEFINED OUTPUT_ROOT)
    if(DEFINED OpenVINO_SOURCE_DIR)
        # For BW compatiblity, when extra modules are built separately
        # but still write its artifacts to OpenVINO source directory
        set(OUTPUT_ROOT ${OpenVINO_SOURCE_DIR})
    else()
        set(OUTPUT_ROOT ${CMAKE_SOURCE_DIR})
    endif()
endif()

# Enable postfixes for Debug/Release builds
set(OV_DEBUG_POSTFIX_WIN "d")
set(OV_RELEASE_POSTFIX_WIN "")
set(OV_DEBUG_POSTFIX_LIN "")
set(OV_RELEASE_POSTFIX_LIN "")
set(OV_DEBUG_POSTFIX_MAC "d")
set(OV_RELEASE_POSTFIX_MAC "")

if(WIN32)
    set(OV_DEBUG_POSTFIX ${OV_DEBUG_POSTFIX_WIN})
    set(OV_RELEASE_POSTFIX ${OV_RELEASE_POSTFIX_WIN})
elseif(APPLE)
    set(OV_DEBUG_POSTFIX ${OV_DEBUG_POSTFIX_MAC})
    set(OV_RELEASE_POSTFIX ${OV_RELEASE_POSTFIX_MAC})
else()
    set(OV_DEBUG_POSTFIX ${OV_DEBUG_POSTFIX_LIN})
    set(OV_RELEASE_POSTFIX ${OV_RELEASE_POSTFIX_LIN})
endif()

set(CMAKE_DEBUG_POSTFIX ${OV_DEBUG_POSTFIX})
set(CMAKE_RELEASE_POSTFIX ${OV_RELEASE_POSTFIX})

# Support CMake multi-configuration for Visual Studio / Ninja or Xcode build
if(OV_GENERATOR_MULTI_CONFIG)
    set(OV_BUILD_POSTFIX $<$<CONFIG:Debug>:${OV_DEBUG_POSTFIX}>$<$<CONFIG:Release>:${OV_RELEASE_POSTFIX}>)
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OV_BUILD_POSTFIX ${OV_DEBUG_POSTFIX})
    else()
        set(OV_BUILD_POSTFIX ${OV_RELEASE_POSTFIX})
    endif()
endif()
add_definitions(-DOV_BUILD_POSTFIX=\"${OV_BUILD_POSTFIX}\")

ov_set_if_not_defined(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})

include(packaging/packaging)

if(APPLE)
    # WA for Xcode generator + object libraries issue:
    # https://gitlab.kitware.com/cmake/cmake/issues/20260
    # http://cmake.3232098.n2.nabble.com/XCODE-DEPEND-HELPER-make-Deletes-Targets-Before-and-While-They-re-Built-td7598277.html
    set(CMAKE_XCODE_GENERATE_TOP_LEVEL_PROJECT_ONLY ON)
endif()

# Use solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# CMake 3.0+: Enable CMAKE_<LANG>_COMPILER_ID AppleClang
set(CMAKE_POLICY_DEFAULT_CMP0025 NEW)
# CMake 3.0+: MACOSX_RPATH is enabled by default.
set(CMAKE_POLICY_DEFAULT_CMP0026 NEW)
# CMake 3.0+ (2.8.12): MacOS "@rpath" in target's install name
set(CMAKE_POLICY_DEFAULT_CMP0042 NEW)
# CMake 3.1+: Simplify variable reference and escape sequence evaluation.
set(CMAKE_POLICY_DEFAULT_CMP0053 NEW)
# CMake 3.3+: Honor visibility properties for all target types
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
# CMake 3.9+: `RPATH` settings on macOS do not affect `install_name`.
set(CMAKE_POLICY_DEFAULT_CMP0068 NEW)
# CMake 3.12+: find_package() uses <PackageName>_ROOT variables.
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
# CMake 3.13+: option() honors normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# CMake 3.15+: export(PACKAGE) does not populate package registry by default.
set(CMAKE_POLICY_DEFAULT_CMP0090 NEW)
# CMake 3.15+: Modules FindPython3, FindPython2 and FindPython use LOCATION for lookup strategy
set(CMAKE_POLICY_DEFAULT_CMP0094 NEW)
# CMake 3.19+: An imported target missing its location property fails during generation.
set(CMAKE_POLICY_DEFAULT_CMP0111 NEW)
# CMake 3.22+: cmake_dependent_option() supports full Condition Syntax
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)
# CMake 3.24+: prefers to set the timestamps of all extracted contents to the time of the extraction
set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
# CMake 3.27+: Visual Studio Generators select latest Windows SDK by default.
set(CMAKE_POLICY_DEFAULT_CMP0149 NEW)
# CMake 3.31+: install() DESTINATION paths are normalized.
set(CMAKE_POLICY_DEFAULT_CMP0177 NEW)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF CACHE BOOL "Disables search in user / system package registries")
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON CACHE BOOL "Disables search in user package registries")
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON CACHE BOOL "Disables search in system package registries")
set(CMAKE_EXPORT_PACKAGE_REGISTRY OFF CACHE BOOL "Disables package registry. Required for 3rd party projects like rapidjson, gflags")
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON CACHE BOOL "Disables package registry. Required for 3rd party projects like rapidjson, gflags")
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Don't warn about obsolete cmake versions in 3rdparty")
set(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION ON CACHE BOOL "Warn about absolute paths in destination")

# LTO

if(ENABLE_LTO)
    set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    include(CheckIPOSupported)

    check_ipo_supported(RESULT IPO_SUPPORTED
                        OUTPUT OUTPUT_MESSAGE
                        LANGUAGES C CXX)

    if(NOT IPO_SUPPORTED)
        set(ENABLE_LTO "OFF" CACHE STRING "Enable Link Time Optimization" FORCE)
        message(WARNING "IPO / LTO is not supported: ${OUTPUT_MESSAGE}")
    endif()
endif()

# General flags

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include(compile_flags/sdl)
include(compile_flags/os_flags)
include(compile_flags/sanitizer)
include(compile_flags/fuzzing)
include(download/dependency_solver)
include(cross_compile/cross_compiled_func)
include(faster_build)
include(whole_archive)
include(linux_name)
include(models)
include(api_validator/api_validator)

include(vs_version/vs_version)
include(plugins/plugins)
include(frontends/frontends)
include(add_target_helpers)
include(CMakePackageConfigHelpers)

if(ENABLE_FUZZING)
    enable_fuzzing()
endif()

get_linux_name(LINUX_OS_NAME)

# macro to mark target as conditionally compiled

function(ov_mark_target_as_cc TARGET_NAME)
    set(cc_library openvino::conditional_compilation)
    if(TARGET IE::conditional_compilation)
        set(cc_library IE::conditional_compilation)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE ${cc_library})

    if(NOT SELECTIVE_BUILD STREQUAL "ON")
        return()
    endif()

    if(NOT TARGET ${TARGET_NAME})
        message(FATAL_ERROR "${TARGET_NAME} does not represent target")
    endif()

    get_target_property(sources ${TARGET_NAME} SOURCES)
    set_source_files_properties(${sources} PROPERTIES OBJECT_DEPENDS ${GENERATED_HEADER})
    add_dependencies(${TARGET_NAME} conditional_compilation_gen)
endfunction()

include(python_requirements)

# Code style utils

include(cpplint/cpplint)
include(clang_format/clang_format)
include(clang_tidy/clang_tidy)
include(ncc_naming_style/ncc_naming_style)

# Restore state
set(CMAKE_MODULE_PATH ${OLD_CMAKE_MODULE_PATH})
