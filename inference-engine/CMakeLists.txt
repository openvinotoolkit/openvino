# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(InferenceEngine)

set(DEV_BUILD TRUE)

include(CTest)

## WA for problem with gtest submodule. It cannot detect uint32 type.
## remove Gtest submodule and this two lines together
include (CheckTypeSize)
check_type_size (uint32_t uint32_t LANGUAGE CXX)

if (UNIX AND NOT APPLE)
set(LINUX TRUE)
endif()

option (OS_FOLDER "create OS dedicated folder in output" OFF)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7l")
    set (ARCH_FOLDER armv7l)
elseif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set (ARCH_FOLDER intel64)
else()
    set (ARCH_FOLDER  ia32)
endif()

if (OS_FOLDER)
	message ("**** OS FOLDER IS: [${OS_FOLDER}]")
	if ("${OS_FOLDER}" STREQUAL "ON")
		message ("**** USING OS FOLDER: [${CMAKE_SYSTEM_NAME}]")
		set (BIN_FOLDER bin/${CMAKE_SYSTEM_NAME}/${ARCH_FOLDER})
	else()
		set (BIN_FOLDER bin/${OS_FOLDER}/${ARCH_FOLDER})
	endif()
else()
    set (BIN_FOLDER bin/${ARCH_FOLDER})
endif()

set (IE_MAIN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (CMAKE_MODULE_PATH "${IE_MAIN_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#printing debug messages
include (debug)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    debug_message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()
message(STATUS "BUILD_CONFIGURATION: ${CMAKE_BUILD_TYPE}")

if(COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0")
endif()

if (UNIX)
    SET(LIB_DL ${CMAKE_DL_LIBS})
endif()

set (OUTPUT_ROOT ${IE_MAIN_SOURCE_DIR})

include(os_flags)

#resolving dependencies for the project
include (dependencies)

set(CMAKE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX})
set(CMAKE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX})

if (WIN32)
    # Support CMake multiconfiguration for Visual Studio build
    set(IE_BUILD_POSTFIX $<$<CONFIG:Debug>:${IE_DEBUG_POSTFIX}>$<$<CONFIG:Release>:${IE_RELEASE_POSTFIX}>)
    set(IE_BUILD_CONFIGURATION $<CONFIG>)
else ()
    if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
        set(IE_BUILD_POSTFIX ${IE_DEBUG_POSTFIX})
    else()
        set(IE_BUILD_POSTFIX ${IE_RELEASE_POSTFIX})
    endif()
    set(IE_BUILD_CONFIGURATION ${CMAKE_BUILD_TYPE})
endif()

add_definitions(-DIE_BUILD_POSTFIX=\"${IE_BUILD_POSTFIX}\")

if(NOT(UNIX))
    if (WIN32)
        #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
        #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
    endif()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (CMAKE_LIBRARY_PATH ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION})
    set (LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

#rpath fully disabled
if (NOT ENABLE_PLUGIN_RPATH)
    SET (CMAKE_SKIP_RPATH TRUE)
endif()

#Use solution folders.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#message("=====================> ${CMAKE_BUILD_TYPE} <=====================")

message (STATUS "PROJECT ............................... " ${PROJECT_NAME})
message (STATUS "CMAKE_BINARY_DIR ...................... " ${CMAKE_BINARY_DIR})
message (STATUS "IE_MAIN_SOURCE_DIR .................... " ${IE_MAIN_SOURCE_DIR})
message (STATUS "CMAKE_GENERATOR ....................... " ${CMAKE_GENERATOR})
message (STATUS "CMAKE_C_COMPILER_ID ................... " ${CMAKE_C_COMPILER_ID})

include(sdl)

set (CMAKE_POSITION_INDEPENDENT_CODE ON)

include (sanitizer)

include(CheckCXXCompilerFlag)

include(cpplint)

add_subdirectory(src)
if (NOT DISABLE_IE_TESTS)
    add_subdirectory(tests)
endif()
add_subdirectory(thirdparty)
set(InferenceEngine_DIR "${CMAKE_BINARY_DIR}")

#to be able to link
set (LIB_FOLDER ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${IE_BUILD_CONFIGURATION}/lib)

# gflags and format_reader targets are kept inside of samples directory and
# they must be built even if samples build is disabled (required for tests and tools).
add_subdirectory(samples)

file(GLOB_RECURSE SAMPLES_SOURCES samples/*.cpp samples/*.hpp samples/*.h)
add_cpplint_target(sample_cpplint
    FOR_SOURCES ${SAMPLES_SOURCES}
    EXCLUDE_PATTERNS "thirdparty/*" "pugixml/*")

if (ENABLE_PYTHON)
    add_subdirectory(ie_bridges/python)
endif()

add_cpplint_report_target()
