# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.13)

if(NOT DEFINED IEDevScripts_DIR)
    message(FATAL_ERROR "IEDevScripts_DIR is not defined")
endif()

macro(ov_set_if_not_defined var value)
    if(NOT DEFINED ${var})
        set(${var} ${value})
    endif()
endmacro()

set(OLD_CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "${IEDevScripts_DIR}")

function(set_ci_build_number)
    set(repo_root "${CMAKE_SOURCE_DIR}")
    include(version)
    foreach(var CI_BUILD_NUMBER OpenVINO_VERSION OpenVINO_SOVERSION OpenVINO_VERSION_SUFFIX OpenVINO_VERSION_BUILD
                OpenVINO_VERSION_MAJOR OpenVINO_VERSION_MINOR OpenVINO_VERSION_PATCH)
        if(NOT DEFINED ${var})
            message(FATAL_ERROR "${var} version component is not defined")
        endif()
        set(${var} "${${var}}" PARENT_SCOPE)
    endforeach()
endfunction()

include(features)

set_ci_build_number()

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

function(set_temp_directory temp_variable source_tree_dir)
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

# Search packages for the host system instead of packages for the target system
# in case of cross compilation these macros should be defined by the toolchain file

if(CMAKE_CROSSCOMPILING AND NOT (OV_ARCH STREQUAL OV_HOST_ARCH AND
                                 CMAKE_SYSTEM_NAME STREQUAL CMAKE_HOST_SYSTEM_NAME))
    # don't look at directories which are part of PATH (with removed bin / sbin at the end)
    # like /opt/homebrew on macOS where we cannot use system env path, because brew's
    # dependencies will be found, but at the same time we need to find flatbufffers and
    # other build system dependencies
    ov_set_if_not_defined(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)
    ov_set_if_not_defined(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY OFF)
    # it contains /usr and if we set this var to OFF, then CMAKE_FIND_ROOT_PATH is ignored
    # ov_set_if_not_defined(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH OFF)
    ov_set_if_not_defined(CMAKE_SYSTEM_PREFIX_PATH OFF)
    ov_set_if_not_defined(CMAKE_SYSTEM_INCLUDE_PATH OFF)
    ov_set_if_not_defined(CMAKE_SYSTEM_LIBRARY_PATH OFF)
    ov_set_if_not_defined(CMAKE_SYSTEM_PROGRAM_PATH OFF)
    if(LINUX)
        # set root paths (overridden to /usr/lib/<CMAKE_LIBRARY_ARCHITECTURE>/cmake)
        # CMAKE_LIBRARY_ARCHITECTURE is defined automatically by cmake after trying the compilers
        ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH "/usr")
    endif()

    # controling CMAKE_FIND_ROOT_PATH usage
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
    ov_set_if_not_defined(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
endif()

macro(__ov_cmake_find_system_path_save_and_reset)
    foreach(v
            CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH
            CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY
            CMAKE_FIND_USE_CMAKE_SYSTEM_PATH
            CMAKE_SYSTEM_PREFIX_PATH
            CMAKE_SYSTEM_INCLUDE_PATH
            CMAKE_SYSTEM_LIBRARY_PATH
            CMAKE_SYSTEM_PROGRAM_PATH
            )
        set(__ov_save_${v} ${${v}})
        set(${v} ON)
    endforeach()
endmacro()

macro(__ov_cmake_find_system_path_restore)
    foreach(v
            CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH
            CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY
            CMAKE_FIND_USE_CMAKE_SYSTEM_PATH
            CMAKE_SYSTEM_PREFIX_PATH
            CMAKE_SYSTEM_INCLUDE_PATH
            CMAKE_SYSTEM_LIBRARY_PATH
            CMAKE_SYSTEM_PROGRAM_PATH
            )
        set(${v} ${__ov_save_${v}})
        unset(__ov_save_${v})
    endforeach()
endmacro()

macro(__ov_cmake_find_root_save_and_reset)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(__ov_save_${v} ${${v}})
        set(${v} NEVER)
    endforeach()
endmacro()

macro(__ov_cmake_find_root_restore)
    foreach(v
            CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
            CMAKE_FIND_ROOT_PATH_MODE_INCLUDE
            CMAKE_FIND_ROOT_PATH_MODE_PACKAGE
            CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
            )
        set(${v} ${__ov_save_${v}})
        unset(__ov_save_${v})
    endforeach()
endmacro()

macro(__ov_cmake_target_flags_save_and_reset)
    foreach(v WIN32 UNIX LINUX APPLE ANDROID BSD)
        set(__ov_target_save_${v} ${${v}})
        unset(${v})
    endforeach()

    if(CMAKE_HOST_WIN32)
        set(WIN32 1)
    elseif(CMAKE_HOST_APPLE)
        set(APPLE 1)
        set(UNIX 1)
    elseif(CMAKE_HOST_LINUX)
        set(LINUX 1)
        set(UNIX 1)
    elseif(CMAKE_HOST_UNIX)
        set(UNIX 1)
    elseif(CMAKE_HOST_BSD)
        set(BSD 1)
    endif()
endmacro()

macro(__ov_cmake_target_flags_restore)
    foreach(v WIN32 UNIX LINUX APPLE ANDROID BSD)
        set(${v} ${__ov_target_save_${v}})
        unset(__ov_target_save_${v})
    endforeach()
endmacro()

if(CMAKE_CROSSCOMPILING)
    # macro to find programs on the host OS
    if(NOT COMMAND find_host_package)
        macro(find_host_package)
            __ov_cmake_find_root_save_and_reset()
            __ov_cmake_target_flags_save_and_reset()
            __ov_cmake_find_system_path_save_and_reset()
            find_package(${ARGN})
            __ov_cmake_find_system_path_restore()
            __ov_cmake_target_flags_restore()
            __ov_cmake_find_root_restore()
        endmacro()
    endif()
    if(NOT COMMAND find_host_program)
        macro(find_host_program)
            __ov_cmake_find_root_save_and_reset()
            __ov_cmake_target_flags_save_and_reset()
            __ov_cmake_find_system_path_save_and_reset()
            find_program(${ARGN})
            __ov_cmake_find_system_path_restore()
            __ov_cmake_target_flags_restore()
            __ov_cmake_find_root_restore()
        endmacro()
    endif()
    if(NOT COMMAND find_host_library)
        macro(find_host_library)
            __ov_cmake_find_root_save_and_reset()
            __ov_cmake_target_flags_save_and_reset()
            __ov_cmake_find_system_path_save_and_reset()
            find_library(${ARGN})
            __ov_cmake_find_system_path_restore()
            __ov_cmake_target_flags_restore()
            __ov_cmake_find_root_restore()
        endmacro()
    endif()
else()
    if(NOT COMMAND find_host_package)
        macro(find_host_package)
            find_package(${ARGN})
        endmacro()
    endif()
    if(NOT COMMAND find_host_program)
        macro(find_host_program)
            find_program(${ARGN})
        endmacro()
    endif()
    if(NOT COMMAND find_host_library)
        macro(find_host_library)
            find_library(${ARGN})
        endmacro()
    endif()
endif()

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
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release;Debug;RelWithDebInfo;MinSizeRel")
endif()

if(USE_BUILD_TYPE_SUBFOLDER)
    set(BIN_FOLDER "${BIN_FOLDER}/${CMAKE_BUILD_TYPE}")
endif()

# allow to override default OUTPUT_ROOT root
if(NOT DEFINED OUTPUT_ROOT)
    if(NOT DEFINED OpenVINO_SOURCE_DIR)
        message(FATAL_ERROR "OpenVINO_SOURCE_DIR is not defined")
    endif()
    set(OUTPUT_ROOT ${OpenVINO_SOURCE_DIR})
endif()

# Enable postfixes for Debug/Release builds
set(IE_DEBUG_POSTFIX_WIN "d")
set(IE_RELEASE_POSTFIX_WIN "")
set(IE_DEBUG_POSTFIX_LIN "")
set(IE_RELEASE_POSTFIX_LIN "")
set(IE_DEBUG_POSTFIX_MAC "d")
set(IE_RELEASE_POSTFIX_MAC "")

if(WIN32)
    set(IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_WIN})
    set(IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_WIN})
elseif(APPLE)
    set(IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_MAC})
    set(IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_MAC})
else()
    set(IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_LIN})
    set(IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_LIN})
endif()

set(CMAKE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX})
set(CMAKE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX})

# Support CMake multi-configuration for Visual Studio / Ninja or Xcode build
if(OV_GENERATOR_MULTI_CONFIG)
    set(IE_BUILD_POSTFIX $<$<CONFIG:Debug>:${IE_DEBUG_POSTFIX}>$<$<CONFIG:Release>:${IE_RELEASE_POSTFIX}>)
else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(IE_BUILD_POSTFIX ${IE_DEBUG_POSTFIX})
    else()
        set(IE_BUILD_POSTFIX ${IE_RELEASE_POSTFIX})
    endif()
endif()
add_definitions(-DIE_BUILD_POSTFIX=\"${IE_BUILD_POSTFIX}\")

ov_set_if_not_defined(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
ov_set_if_not_defined(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})

if(CPACK_GENERATOR MATCHES "^(DEB|RPM)$")
    # to make sure that lib/<multiarch-tuple> is created on Debian
    set(CMAKE_INSTALL_PREFIX "/usr" CACHE PATH "Cmake install prefix" FORCE)
endif()

include(packaging/packaging)

set(CMAKE_SKIP_INSTALL_RPATH ON)

if(APPLE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

    if(DEFINED OV_CPACK_LIBRARYDIR)
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${OV_CPACK_LIBRARYDIR}")
    else()
        message(FATAL_ERROR "Internal error: OV_CPACK_LIBRARYDIR is not defined, while it's required to initialize RPATH")
    endif()

    if(CPACK_GENERATOR STREQUAL "BREW")
        set(CMAKE_SKIP_INSTALL_RPATH OFF)
    endif()

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
# CMake 3.9+: `RPATH` settings on macOS do not affect `install_name`.
set(CMAKE_POLICY_DEFAULT_CMP0068 NEW)
# CMake 3.12+: find_package() uses <PackageName>_ROOT variables.
set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
# CMake 3.13+: option() honors normal variables.
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
# CMake 3.19+: An imported target missing its location property fails during generation.
set(CMAKE_POLICY_DEFAULT_CMP0111 NEW)
# CMake 3.22+ :cmake_dependent_option() supports full Condition Syntax
set(CMAKE_POLICY_DEFAULT_CMP0127 NEW)

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

macro(ov_install_static_lib target comp)
    if(NOT BUILD_SHARED_LIBS)
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "STATIC_LIBRARY")
            set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL OFF)
        endif()
        install(TARGETS ${target} EXPORT OpenVINOTargets
                ARCHIVE DESTINATION ${OV_CPACK_ARCHIVEDIR} COMPONENT ${comp} ${ARGN})
    endif()
endmacro()

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
include(add_ie_target)
include(CMakePackageConfigHelpers)

if(ENABLE_FUZZING)
    enable_fuzzing()
endif()

get_linux_name(LINUX_OS_NAME)

# macro to mark target as conditionally compiled

function(ie_mark_target_as_cc TARGET_NAME)
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

function(ov_mark_target_as_cc)
    ie_mark_target_as_cc(${ARGN})
endfunction()

include(python_requirements)
include(native_compile)

# Code style utils

include(cpplint/cpplint)
include(clang_format/clang_format)
include(ncc_naming_style/ncc_naming_style)

# Restore state
set(CMAKE_MODULE_PATH ${OLD_CMAKE_MODULE_PATH})
