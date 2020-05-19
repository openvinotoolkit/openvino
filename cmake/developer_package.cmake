# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

list(APPEND CMAKE_MODULE_PATH
        "${OpenVINO_MAIN_SOURCE_DIR}/cmake/download"
        "${OpenVINO_MAIN_SOURCE_DIR}/cmake/cross_compile"
        )

include(CPackComponent)
unset(IE_CPACK_COMPONENTS_ALL CACHE)

set(IE_CPACK_IE_DIR       deployment_tools/inference_engine)

# Search packages for the host system instead of packages for the target system
# in case of cross compilation these macros should be defined by the toolchain file
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

#
# ie_cpack_set_library_dir()
#
# Set library directory for cpack
#
function(ie_cpack_set_library_dir)
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH}/${CMAKE_BUILD_TYPE} PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH ${IE_CPACK_IE_DIR}/bin/${ARCH}/${CMAKE_BUILD_TYPE} PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH}/${CMAKE_BUILD_TYPE} PARENT_SCOPE)
    else()
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH} PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH} PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH} PARENT_SCOPE)
    endif()
endfunction()

ie_cpack_set_library_dir()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
macro(ie_cpack_add_component NAME)
    list(APPEND IE_CPACK_COMPONENTS_ALL ${NAME})
    set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)
    cpack_add_component(${NAME} ${ARGN})
endmacro()

macro(ie_cpack)
    set(CPACK_GENERATOR "TGZ")
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_${CMAKE_BUILD_TYPE})
        string(REPLACE "\\" "_" CPACK_PACKAGE_VERSION "${CI_BUILD_NUMBER}")
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
        string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "${CI_BUILD_NUMBER}")
    endif()
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_PACKAGE_VENDOR "Intel")
    set(CPACK_COMPONENTS_ALL ${ARGN})
    set(CPACK_STRIP_FILES ON)

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    include(CPack)
endmacro()

# prepare temporary folder
function(set_temp_directory temp_variable source_tree_dir)
    if (DEFINED ENV{DL_SDK_TEMP} AND NOT $ENV{DL_SDK_TEMP} STREQUAL "")
        message(STATUS "DL_SDK_TEMP environment is set : $ENV{DL_SDK_TEMP}")

        if (WIN32)
            string(REPLACE "\\" "\\\\" temp $ENV{DL_SDK_TEMP})
        else()
            set(temp $ENV{DL_SDK_TEMP})
        endif()

        if (ENABLE_ALTERNATIVE_TEMP)
            set(ALTERNATIVE_PATH ${source_tree_dir}/temp)
        endif()
    else ()
        set(temp ${source_tree_dir}/temp)
    endif()

    set("${temp_variable}" "${temp}" CACHE PATH "Path to temp directory")
    if(ALTERNATIVE_PATH)
        set(ALTERNATIVE_PATH "${ALTERNATIVE_PATH}" PARENT_SCOPE)
    endif()
endfunction()

include(coverage/coverage)

# External dependencies
find_package(Threads)

# Detect target
include(target_flags)

# printing debug messages
include(debug)

# linking libraries without discarding symbols
include(whole_archive)

string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH_FOLDER)
if(ARCH_FOLDER STREQUAL "x86_64" OR ARCH_FOLDER STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
    set(ARCH_FOLDER intel64)
elseif(ARCH_FOLDER STREQUAL "i386")
    set(ARCH_FOLDER ia32)
endif()

if(OS_FOLDER)
	message ("**** OS FOLDER IS: [${OS_FOLDER}]")
	if("${OS_FOLDER}" STREQUAL "ON")
		message ("**** USING OS FOLDER: [${CMAKE_SYSTEM_NAME}]")
		set(BIN_FOLDER "bin/${CMAKE_SYSTEM_NAME}/${ARCH_FOLDER}")
	else()
		set(BIN_FOLDER "bin/${OS_FOLDER}/${ARCH_FOLDER}")
	endif()
else()
    set(BIN_FOLDER "bin/${ARCH_FOLDER}")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    debug_message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# allow to override default OUTPUT_ROOT root
if(NOT DEFINED OUTPUT_ROOT)
    set(OUTPUT_ROOT ${OpenVINO_MAIN_SOURCE_DIR})
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

if (WIN32)
    # Support CMake multiconfiguration for Visual Studio build
    set(IE_BUILD_POSTFIX $<$<CONFIG:Debug>:${IE_DEBUG_POSTFIX}>$<$<CONFIG:Release>:${IE_RELEASE_POSTFIX}>)
else ()
    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
        set(IE_BUILD_POSTFIX ${IE_DEBUG_POSTFIX})
    else()
        set(IE_BUILD_POSTFIX ${IE_RELEASE_POSTFIX})
    endif()
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_definitions(-DIE_BUILD_POSTFIX=\"${IE_BUILD_POSTFIX}\")

if(NOT UNIX)
    if (WIN32)
        # set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    endif()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

# Use solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_POLICY_DEFAULT_CMP0054 NEW)

include(sdl)
include(os_flags)
include(sanitizer)
include(cross_compiled_func)

function(set_ci_build_number)
    set(OpenVINO_MAIN_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    include(version)
    set(CI_BUILD_NUMBER "${CI_BUILD_NUMBER}" PARENT_SCOPE)
endfunction()
set_ci_build_number()
