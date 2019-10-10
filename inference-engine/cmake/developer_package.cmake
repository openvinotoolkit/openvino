# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# printing debug messages
include(debug)

if (UNIX AND NOT APPLE)
    set(LINUX ON)
endif()

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

if(COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -O0")
endif()

if(UNIX)
    SET(LIB_DL ${CMAKE_DL_LIBS})
endif()

set(OUTPUT_ROOT ${IE_MAIN_SOURCE_DIR})

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
    set(CMAKE_LIBRARY_PATH ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER})
    set(LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}) # compatibility issue: linux uses LIBRARY_OUTPUT_PATH, windows uses LIBRARY_OUTPUT_DIRECTORY
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(CMAKE_PDB_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set(LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set(LIBRARY_OUTPUT_PATH ${LIBRARY_OUTPUT_DIRECTORY}/lib)
endif()

if(APPLE)
	set(CMAKE_MACOSX_RPATH 1)
endif(APPLE)

# rpath fully disabled
if (NOT ENABLE_PLUGIN_RPATH)
    set(CMAKE_SKIP_RPATH TRUE)
endif()

# prepare temporary folder
function(set_temp_directory temp_variable source_tree_dir)
    if (DEFINED ENV{${DL_SDK_TEMP}} AND NOT $ENV{${DL_SDK_TEMP}} STREQUAL "")
        if (WIN32)
            string(REPLACE "\\" "\\\\" temp $ENV{${DL_SDK_TEMP}})
        else(WIN32)
            set(temp $ENV{${DL_SDK_TEMP}})
        endif(WIN32)

        if (ENABLE_ALTERNATIVE_TEMP)
            set(ALTERNATIVE_PATH ${source_tree_dir}/temp)
        endif()
    else ()
        message(STATUS "DL_SDK_TEMP envionment not set")
        set(temp ${source_tree_dir}/temp)
    endif()

    set("${temp_variable}" "${temp}" PARENT_SCOPE)
    if(ALTERNATIVE_PATH)
        set(ALTERNATIVE_PATH "${ALTERNATIVE_PATH}" PARENT_SCOPE)
    endif()
endfunction()

# Use solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(os_flags)
include(sdl)
include(sanitizer)
include(cpplint)
include(cppcheck)

function(set_ci_build_number)
    set(IE_MAIN_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
    include(version)
    set(CI_BUILD_NUMBER "${CI_BUILD_NUMBER}" PARENT_SCOPE)
endfunction()
set_ci_build_number()

if(ENABLE_PROFILING_ITT)
    find_package(ITT REQUIRED)
endif()

include(plugins/plugins)
