# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8.12)

project(Samples)

if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE not defined, 'Release' will be used")
    set(CMAKE_BUILD_TYPE "Release")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT(BIN_FOLDER))
    string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} ARCH)
    if(ARCH STREQUAL "x86_64" OR ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
        set(ARCH intel64)
    elseif(ARCH STREQUAL "i386")
        set(ARCH ia32)
    endif()

    set (BIN_FOLDER ${ARCH})
endif()

if(IE_MAIN_SOURCE_DIR)
    # in case if samples are built from IE repo
    set(IE_MAIN_SAMPLES_DIR ${OpenVINO_MAIN_SOURCE_DIR})
    # hint for find_package(InferenceEngine in the samples folder)
    set(InferenceEngine_DIR "${CMAKE_BINARY_DIR}")
else()
    # in case if samples are built out of IE repo
    set(IE_MAIN_SAMPLES_DIR ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(NOT(UNIX))
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER})
else ()
    set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/lib)
    set (CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_PDB_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
    set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${IE_MAIN_SAMPLES_DIR}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE})
endif()

if (WIN32)
    if (NOT "${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
        message(FATAL_ERROR "Only 64-bit supported on Windows")
    endif()

    set_property (DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS -DNOMINMAX")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX") #treating warnings as errors
    endif ()

    if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251 /wd4275 /wd4267") #disable some warnings
    endif()
else()
    if(TREAT_WARNING_AS_ERROR)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror") #treating warnings as errors
    endif()

    if (APPLE)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
    elseif(UNIX)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL Clang)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
        endif()
    endif()
endif()

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

####################################
## to use C++11; can overwritten via cmake command line
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_EXTENSIONS OFF)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
    if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    endif()
endif()
####################################

set (GFLAGS_IS_SUBPROJECT TRUE)
set (HAVE_SYS_STAT_H 1)
set (HAVE_INTTYPES_H 1)
set (INTTYPES_FORMAT C99)
set (BUILD_TESTING OFF)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gflags")
    add_subdirectory(thirdparty/gflags)
    set_target_properties(gflags_nothreads_static PROPERTIES FOLDER thirdparty)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

if(IE_NOT_FOUND_MESSAGE)
    # the flag is used to throw a custom message in case if the IE package is not found.
    find_package(InferenceEngine 2.1 QUIET)
    if (NOT(InferenceEngine_FOUND))
        message(FATAL_ERROR ${IE_NOT_FOUND_MESSAGE})
    endif()
else()
    find_package(InferenceEngine 2.1 REQUIRED)
endif()

# format reader must be added after find_package(InferenceEngine) to get
# exactly the same OpenCV_DIR path which was used for the InferenceEngine build
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/common/format_reader")
    add_subdirectory(common/format_reader)
endif()

# samples build can be switched off during whole IE build
if (IE_MAIN_SOURCE_DIR AND NOT ENABLE_SAMPLES)
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
#               [OPENCV_DEPENDENCIES <opencv modules>]
#               [EXCLUDE_CPPLINT]
#
macro(ie_add_sample)
    set(options EXCLUDE_CPPLINT)
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES OPENCV_DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    # Find OpenCV components if exist
    if(IE_SAMPLE_OPENCV_DEPENDENCIES)
        find_package(OpenCV COMPONENTS ${IE_SAMPLE_OPENCV_DEPENDENCIES} QUIET)
        if(NOT OpenCV_FOUND)
            message(WARNING "OPENCV is disabled or not found, " ${IE_SAMPLE_NAME} " skipped")
            return()
        endif()
    endif()

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${IE_SAMPLES_SOURCES})
    if(IE_SAMPLES_HEADERS)
        source_group("include" FILES ${IE_SAMPLES_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${IE_SAMPLE_NAME} ${IE_SAMPLE_SOURCES} ${IE_SAMPLES_HEADERS})
    if(IE_SAMPLE_OPENCV_DEPENDENCIES)
        target_compile_definitions(${IE_SAMPLE_NAME} PRIVATE USE_OPENCV)
    endif()

    set(folder_name cpp_samples)
    if(IE_SAMPLE_NAME MATCHES ".*_c$")
        set(folder_name c_samples)
    endif()

    set_target_properties(${IE_SAMPLE_NAME} PROPERTIES FOLDER ${folder_name}
                                                       COMPILE_PDB_NAME ${IE_SAMPLE_NAME})

    if(IE_SAMPLE_INCLUDE_DIRECTORIES)
        target_include_directories(${IE_SAMPLE_NAME} PRIVATE ${IE_SAMPLE_INCLUDE_DIRECTORIES})
    endif()
    target_include_directories(${IE_SAMPLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../common")

    target_link_libraries(${IE_SAMPLE_NAME} PRIVATE ${OpenCV_LIBRARIES} ${InferenceEngine_LIBRARIES}
                                                    ${IE_SAMPLE_DEPENDENCIES} gflags)

    # create global target with all samples / demo apps
    if(NOT TARGET ie_samples)
        add_custom_target(ie_samples ALL)
    endif()
    add_dependencies(ie_samples ${IE_SAMPLE_NAME})

    if(COMMAND add_cpplint_target AND NOT IE_SAMPLE_EXCLUDE_CPPLINT)
        if(folder_name STREQUAL "c_samples")
            set(custom_filters "-readability/casting,-runtime/printf")
        endif()
        add_cpplint_target(${IE_SAMPLE_NAME}_cpplint FOR_TARGETS ${IE_SAMPLE_NAME}
                           CUSTOM_FILTERS ${custom_filters})
    endif()
endmacro()

# collect all samples subdirectories
file(GLOB samples_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
# skip building of unnecessary subdirectories
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")
    list(REMOVE_ITEM samples_dirs common thirdparty)
endif()
add_samples_to_build(${samples_dirs})
