# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required (VERSION 3.3)

project(InferenceEngine)

set(DEV_BUILD TRUE)

## WA for problem with gtest submodule. It cannot detect uint32 type.
## remove Gtest submodule and this two lines together
include (CheckTypeSize)
check_type_size (uint32_t uint32_t LANGUAGE CXX)

if (UNIX AND NOT APPLE)
set(LINUX TRUE)
endif()

option (OS_FOLDER "create OS dedicated folder in output" OFF)

if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
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
else()
endif()

set (OUTPUT_ROOT ${IE_MAIN_SOURCE_DIR})

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
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

include(os_flags)

#resolving rependencies for the project
include (dependencies)

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

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    include(sdl)
endif()
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

include (sanitizer)

include(CheckCXXCompilerFlag)
if(UNIX)
    CHECK_CXX_COMPILER_FLAG("-fvisibility=hidden" COMPILER_SUPPORTS_VISIBILITY)
    if (COMPILER_SUPPORTS_VISIBILITY)
        #add_definitions(-fvisibility=hidden) todo: should be hidden? if so define default visibiliti explicite for each funtion
        add_definitions(-fvisibility=default)
    endif(COMPILER_SUPPORTS_VISIBILITY)
endif(UNIX)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(thirdparty)

option(ENABLE_PYTHON_BINDINGS "Build python bindings." OFF)
if (ENABLE_PYTHON_BINDINGS)
    set(InferenceEngine_DIR "${CMAKE_BINARY_DIR}")

    #to be able to link
    set (LIB_FOLDER ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    add_subdirectory(ie_bridges/python)
endif()

if (ENABLE_SAMPLES_CORE)
    set(InferenceEngine_DIR "${CMAKE_BINARY_DIR}")

    #to be able to link
    set (LIB_FOLDER ${IE_MAIN_SOURCE_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    add_subdirectory(samples)
endif()
