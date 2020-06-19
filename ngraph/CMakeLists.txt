# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

cmake_minimum_required (VERSION 3.11)

# set directory where the custom finders live
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed.")
endif()

if (CMAKE_BUILD_TYPE)
    set(RELEASE_TYPES Debug Release RelWithDebInfo MinSizeRel)
    list(FIND RELEASE_TYPES ${CMAKE_BUILD_TYPE} INDEX_FOUND)
    if (${INDEX_FOUND} EQUAL -1)
        message(FATAL_ERROR "CMAKE_BUILD_TYPE must be one of Debug, Release, RelWithDebInfo, or MinSizeRel")
    endif()
endif()

include(git_tags)
include(var_functions)

NGRAPH_GET_VERSION_LABEL()

string(REGEX MATCH "([0-9?]+)\\.([0-9?]+)\\.([0-9?]+)(-(rc|dev)\\.[0-9?]+)?" NGRAPH_VERSION_SHORT "${NGRAPH_VERSION_LABEL}")
string(REGEX REPLACE "-rc." "rc" NGRAPH_WHEEL_VERSION "${NGRAPH_VERSION_SHORT}")
string(REGEX REPLACE "-dev." "dev" NGRAPH_WHEEL_VERSION "${NGRAPH_WHEEL_VERSION}")
string(REGEX MATCH "([0-9?]+)\\.([0-9?]+)" NGRAPH_API_VERSION "${NGRAPH_VERSION_LABEL}")
string(REGEX MATCH "[^v](.*)" NGRAPH_VERSION "${NGRAPH_VERSION_LABEL}")
string(REPLACE "." ";" NGRAPH_VERSION_PARTS "${NGRAPH_VERSION_SHORT}")
list(GET NGRAPH_VERSION_PARTS 0 NGRAPH_VERSION_MAJOR)
list(GET NGRAPH_VERSION_PARTS 1 NGRAPH_VERSION_MINOR)
list(GET NGRAPH_VERSION_PARTS 2 NGRAPH_VERSION_PATCH)
configure_file(VERSION.in VERSION)

message(STATUS "NGRAPH_VERSION ${NGRAPH_VERSION}")
message(STATUS "NGRAPH_VERSION_SHORT ${NGRAPH_VERSION_SHORT}")
message(STATUS "NGRAPH_WHEEL_VERSION ${NGRAPH_WHEEL_VERSION}")
message(STATUS "NGRAPH_API_VERSION ${NGRAPH_API_VERSION}")

set(NGRAPH_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if (APPLE)
    # Enable MACOS_RPATH by default.
    cmake_policy(SET CMP0042 NEW)
    # Enable CMAKE_<LANG>_COMPILER_ID AppleClang
    cmake_policy(SET CMP0025 NEW)
endif()

project (ngraph)

if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# APPLE: Set CMAKE_OSX_SYSROOT if not set already.
if (APPLE)
    execute_process(COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE OSX_FULL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REGEX REPLACE "^([0-9]+\\.[0-9]+).*$" "\\1"
        OSX_SHORT_VERSION "${OSX_FULL_VERSION}")

    message(STATUS "Detected: OSX ${OSX_SHORT_VERSION}")

    if (CMAKE_OSX_SYSROOT)
        message(STATUS "Using CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT}")
    else()
        execute_process(COMMAND xcode-select -p
            OUTPUT_VARIABLE APPLE_DEV_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        if("${APPLE_DEV_DIR}" STREQUAL "/Library/Developer/CommandLineTools")
            # Command line tools only
            set(XCODE_ISYSROOT ${APPLE_DEV_DIR}/SDKs/MacOSX.sdk)
            message(STATUS "Trying command line tool sdk at ${XCODE_ISYSROOT}.")
            if(NOT EXISTS ${XCODE_ISYSROOT})
                message(FATAL_ERROR "Cannot find macos sdk.")
            endif()
        else()
            # Xcode is installed
            set(XCODE_ISYSROOT ${APPLE_DEV_DIR}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${OSX_SHORT_VERSION}.sdk)
            message(STATUS "Trying Xcode sdk at ${XCODE_ISYSROOT}.")
            if(NOT EXISTS ${XCODE_ISYSROOT})
                set(XCODE_ISYSROOT ${APPLE_DEV_DIR}/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)
                if(NOT EXISTS ${XCODE_ISYSROOT})
                    message(FATAL_ERROR "Cannot find macos sdk.")
                endif()
            endif()
        endif()
        message(STATUS "Setting CMAKE_OSX_SYSROOT for macos ${OSX_SHORT_VERSION} to ${XCODE_ISYSROOT}")
        set(CMAKE_OSX_SYSROOT ${XCODE_ISYSROOT})
    endif()
endif()

option(NGRAPH_UNIT_TEST_ENABLE "Control the building of unit tests" TRUE)
option(NGRAPH_TEST_UTIL_ENABLE "Control the building of test utility" TRUE)
option(NGRAPH_INTERPRETER_ENABLE "Control the building of the INTERPRETER backend" TRUE)
option(NGRAPH_DEBUG_ENABLE "Enable output for NGRAPH_DEBUG statements" FALSE)
option(NGRAPH_DEPRECATED_ENABLE "Enable compiler deprecation pragmas for deprecated APIs (recommended only for development use)" FALSE)
option(NGRAPH_ONNX_IMPORT_ENABLE "Enable ONNX importer" FALSE)
option(NGRAPH_CODE_COVERAGE_ENABLE "Enable code coverage data collection" FALSE)
option(NGRAPH_LIB_VERSIONING_ENABLE "Enable shared library versioning" FALSE)
option(NGRAPH_PYTHON_BUILD_ENABLE "Enable build nGraph python package wheel" FALSE)
option(NGRAPH_IE_ENABLE "Enable the Inference Engine backend" FALSE)
option(NGRAPH_JSON_ENABLE "Enable JSON based serialization and tracing features" TRUE)
option(NGRAPH_DYNAMIC_COMPONENTS_ENABLE "Enable dynamic loading of components" TRUE)
option(NGRAPH_EXPORT_TARGETS_ENABLE "Enable exporting nGraph cmake export targets" TRUE)
option(NGRAPH_WARNINGS_AS_ERRORS "Make all nGraph compile-time warnings into errors" FALSE)
option(NGRAPH_ADDRESS_SANITIZER_ENABLE "Compiles and links with Address Sanitizer" FALSE)
option(NGRAPH_THREAD_SANITIZER_ENABLE "Compiles and links with Thread Sanitizer" FALSE)
option(NGRAPH_UB_SANITIZER_ENABLE "Compiles and links with Undefined Behavior Sanitizer" FALSE)
option(NGRAPH_EVALUATE_ENABLE "Enable compilation of reference implementations for operations" TRUE)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object" FALSE)
endif()

if (NGRAPH_UNIT_TEST_ENABLE)
    if (NOT NGRAPH_TEST_UTIL_ENABLE)
        message(WARNING "Unit test needs test util: Turning NGRAPH_TEST_UTIL_ENABLE ON!")
        set(NGRAPH_TEST_UTIL_ENABLE TRUE)
    endif()
endif()

macro (NORMALIZE_BOOL VAL)
    if (${VAL})
        set(${VAL} ON)
    else()
        set(${VAL} OFF)
    endif()
endmacro()

NORMALIZE_BOOL(NGRAPH_UNIT_TEST_ENABLE)
NORMALIZE_BOOL(NGRAPH_TEST_UTIL_ENABLE)
NORMALIZE_BOOL(NGRAPH_INTERPRETER_ENABLE)
NORMALIZE_BOOL(NGRAPH_DEBUG_ENABLE)
NORMALIZE_BOOL(NGRAPH_DEPRECATED_ENABLE)
NORMALIZE_BOOL(NGRAPH_ONNX_IMPORT_ENABLE)
NORMALIZE_BOOL(NGRAPH_CODE_COVERAGE_ENABLE)
NORMALIZE_BOOL(NGRAPH_LIB_VERSIONING_ENABLE)
NORMALIZE_BOOL(NGRAPH_PYTHON_BUILD_ENABLE)
NORMALIZE_BOOL(NGRAPH_IE_ENABLE)
NORMALIZE_BOOL(NGRAPH_JSON_ENABLE)

NORMALIZE_BOOL(NGRAPH_DYNAMIC_COMPONENTS_ENABLE)
NORMALIZE_BOOL(NGRAPH_EXPORT_TARGETS_ENABLE)
NORMALIZE_BOOL(NGRAPH_WARNINGS_AS_ERRORS)
NORMALIZE_BOOL(NGRAPH_ADDRESS_SANITIZER_ENABLE)
NORMALIZE_BOOL(NGRAPH_THREAD_SANITIZER_ENABLE)
NORMALIZE_BOOL(NGRAPH_UB_SANITIZER_ENABLE)
NORMALIZE_BOOL(NGRAPH_EVALUATE_ENABLE)

message(STATUS "NGRAPH_ADDRESS_SANITIZER_ENABLE:      ${NGRAPH_ADDRESS_SANITIZER_ENABLE}")
message(STATUS "NGRAPH_CODE_COVERAGE_ENABLE:          ${NGRAPH_CODE_COVERAGE_ENABLE}")
message(STATUS "NGRAPH_DEBUG_ENABLE:                  ${NGRAPH_DEBUG_ENABLE}")
message(STATUS "NGRAPH_DEPRECATED_ENABLE:             ${NGRAPH_DEPRECATED_ENABLE}")
message(STATUS "NGRAPH_DYNAMIC_COMPONENTS_ENABLE:     ${NGRAPH_DYNAMIC_COMPONENTS_ENABLE}")
message(STATUS "NGRAPH_EXPORT_TARGETS_ENABLE:         ${NGRAPH_EXPORT_TARGETS_ENABLE}")
message(STATUS "NGRAPH_IE_ENABLE:                     ${NGRAPH_IE_ENABLE}")
message(STATUS "NGRAPH_IE_STATIC_LIB_ENABLE:          ${NGRAPH_IE_STATIC_LIB_ENABLE}")
message(STATUS "NGRAPH_INTERPRETER_ENABLE:            ${NGRAPH_INTERPRETER_ENABLE}")
message(STATUS "NGRAPH_JSON_ENABLE:                   ${NGRAPH_JSON_ENABLE}")
message(STATUS "NGRAPH_LIB_VERSIONING_ENABLE:         ${NGRAPH_LIB_VERSIONING_ENABLE}")
message(STATUS "NGRAPH_ONNX_IMPORT_ENABLE:            ${NGRAPH_ONNX_IMPORT_ENABLE}")
message(STATUS "NGRAPH_PYTHON_BUILD_ENABLE:           ${NGRAPH_PYTHON_BUILD_ENABLE}")
message(STATUS "NGRAPH_TEST_UTIL_ENABLE:              ${NGRAPH_TEST_UTIL_ENABLE}")
message(STATUS "NGRAPH_THREAD_SANITIZER_ENABLE:       ${NGRAPH_THREAD_SANITIZER_ENABLE}")
message(STATUS "NGRAPH_TOOLS_ENABLE:                  ${NGRAPH_TOOLS_ENABLE}")
message(STATUS "NGRAPH_UB_SANITIZER_ENABLE:           ${NGRAPH_UB_SANITIZER_ENABLE}")
message(STATUS "NGRAPH_UNIT_TEST_ENABLE:              ${NGRAPH_UNIT_TEST_ENABLE}")
message(STATUS "NGRAPH_WARNINGS_AS_ERRORS:            ${NGRAPH_WARNINGS_AS_ERRORS}")
message(STATUS "NGRAPH_EVALUATE_ENABLE:               ${NGRAPH_EVALUATE_ENABLE}")

set(NGRAPH_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD ${NGRAPH_CXX_STANDARD})
message(STATUS "NGRAPH_CXX_STANDARD:                  ${NGRAPH_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create compilation database compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Setup CMAKE_ARGS to be forwarded to External Projects
set(NGRAPH_FORWARD_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=${CMAKE_EXPORT_COMPILE_COMMANDS}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}
    )

if(CMAKE_TOOLCHAIN_FILE)
    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            )
endif()

if (CMAKE_OSX_SYSROOT)
    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            )
endif()

if (NOT MSVC)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    endif()

    set(NGRAPH_FORWARD_CMAKE_ARGS
            ${NGRAPH_FORWARD_CMAKE_ARGS}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            )
endif()
message(STATUS "NGRAPH_FORWARD_CMAKE_ARGS ${NGRAPH_FORWARD_CMAKE_ARGS}")

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_BINDIR "bin")
    set(CMAKE_INSTALL_INCLUDEDIR "include")
    set(CMAKE_INSTALL_DOCDIR "doc")
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if (DEFINED NGRAPH_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX ${NGRAPH_INSTALL_PREFIX})
endif()
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")

# Destinations
set(NGRAPH_INSTALL_LIB "${NGRAPH_COMPONENT_PREFIX}${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "${NGRAPH_COMPONENT_PREFIX}${CMAKE_INSTALL_INCLUDEDIR}")
set(NGRAPH_INSTALL_DOC "${NGRAPH_COMPONENT_PREFIX}${CMAKE_INSTALL_DOCDIR}")
set(NGRAPH_INSTALL_BIN "${NGRAPH_COMPONENT_PREFIX}${CMAKE_INSTALL_BINDIR}")
if (LINUX)
    if (DEFINED NGRAPH_RPATH)
        set(CMAKE_BUILD_RPATH "$ORIGIN:${NGRAPH_RPATH}")
        set(CMAKE_INSTALL_RPATH "$ORIGIN:${NGRAPH_RPATH}")
    else()
        set(CMAKE_BUILD_RPATH "$ORIGIN")
        set(CMAKE_INSTALL_RPATH "$ORIGIN")
    endif()
endif()

#-----------------------------------------------------------------------------------------------
# Compile Flags to pass to external projects...
#-----------------------------------------------------------------------------------------------

include(cmake/sdl.cmake)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (DEFINED NGRAPH_USE_CXX_ABI)
        message( STATUS "nGraph using CXX11 ABI: " ${NGRAPH_USE_CXX_ABI} )
        list(APPEND NGRAPH_COMMON_FLAGS -D_GLIBCXX_USE_CXX11_ABI=${NGRAPH_USE_CXX_ABI})
    endif()
endif()

string(REPLACE " " ";" NGRAPH_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " " ";" NGRAPH_CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
list(APPEND NGRAPH_COMMON_CXX_FLAGS ${NGRAPH_CMAKE_CXX_FLAGS} ${NGRAPH_COMMON_FLAGS})
list(APPEND NGRAPH_COMMON_C_FLAGS ${NGRAPH_CMAKE_C_FLAGS} ${NGRAPH_COMMON_FLAGS})
message(STATUS "NGRAPH_COMMON_CXX_FLAGS ${NGRAPH_COMMON_CXX_FLAGS}")
message(STATUS "NGRAPH_COMMON_C_FLAGS ${NGRAPH_COMMON_C_FLAGS}")

# Set baseline flags for nGraph source files
string(REPLACE ";" " " CMAKE_ORIGINAL_CXX_FLAGS "${NGRAPH_COMMON_CXX_FLAGS}")

#-----------------------------------------------------------------------------------------------
# Compile Flags for nGraph...
#-----------------------------------------------------------------------------------------------

# Prevent Eigen from using any LGPL3 code
set(CMAKE_CXX_FLAGS "${CMAKE_ORIGINAL_CXX_FLAGS} -DEIGEN_MPL2_ONLY -DTBB_USE_THREADING_TOOLS")

if (WIN32)
    string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

# Compiler-specific logic...
if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP")

    if (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
        # C4251 needs to have dll-interface to be used by clients of class
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4251")
        # C4275 non dll-interface class used as base for dll-interface class
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4275")
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g")
    set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")
    set(CMAKE_CXX_VISIBILITY_PRESET default)
    set(CMAKE_C_VISIBILITY_PRESET default)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)
    endif()

if (NOT WIN32 AND ${NGRAPH_WARNINGS_AS_ERRORS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if (NGRAPH_CODE_COVERAGE_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

if (DEFINED NGRAPH_TUNE_ARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=${NGRAPH_TUNE_ARCH}")
endif()

if (NGRAPH_USE_GOLD)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
endif()

if (NGRAPH_IE_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_IE_ENABLE")
endif()

if (NGRAPH_DYNAMIC_COMPONENTS_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE")
endif()

if(NGRAPH_ADDRESS_SANITIZER_ENABLE)
    if (APPLE)
        set(ADDRESS_SANITIZER_FLAGS "-fsanitize=address")
    else()
        set(ADDRESS_SANITIZER_FLAGS "-fsanitize=address -fsanitize=leak")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
endif()

if(NGRAPH_THREAD_SANITIZER_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
endif()

if(NGRAPH_UB_SANITIZER_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
endif()

if(NGRAPH_ADDRESS_SANITIZER_ENABLE OR NGRAPH_THREAD_SANITIZER_ENABLE OR NGRAPH_UB_SANITIZER_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
endif()

# Since UNIX support Bash we can use a Bash script to do the clang-format functions
# This is much faster than the cmake method
if (UNIX)
    add_custom_target(style-check COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/maint/check-code-format.sh)
    add_custom_target(style-apply COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/maint/apply-code-format.sh)
    add_custom_target(style COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/maint/apply-code-format.sh)
else()
    add_custom_target(style-check
        COMMAND ${CMAKE_COMMAND}
        -DNGRAPH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/style_check.cmake
    )

    add_custom_target(style-apply
        COMMAND ${CMAKE_COMMAND}
        -DNGRAPH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/style_apply.cmake
    )

    add_custom_target(style
        COMMAND ${CMAKE_COMMAND}
        -DNGRAPH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/style_apply.cmake
    )
endif()

add_custom_target(fix-mode
    COMMAND ${CMAKE_COMMAND}
    -DNGRAPH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/fix_mode.cmake
)

if (NGRAPH_EVALUATE_ENABLE)
    add_definitions(-DNGRAPH_EVALUATE_ENABLE)
endif()

#-----------------------------------------------------------------------------------------------
# enable or disable output from NGRAPH_DEBUG statements
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

#-----------------------------------------------------------------------------------------------
# enable or disable deprecation warnings for old APIs
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEPRECATED_ENABLE)
    add_definitions(-DNGRAPH_DEPRECATED_ENABLE)
endif()

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

#-----------------------------------------------------------------------------------------------
# Print Global Options
#-----------------------------------------------------------------------------------------------
message(STATUS "Compile Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Shared Link Flags: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

#-----------------------------------------------------------------------------------------------
# External projects install directory
#-----------------------------------------------------------------------------------------------

if (NOT NGRAPH_BUILD_DIR)
    set(NGRAPH_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/ngraph)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
    if(WIN32)
        set(NGRAPH_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR})
        set(NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY ${NGRAPH_BUILD_DIR}/\${CMAKE_INSTALL_CONFIG_NAME})
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
        set(NGRAPH_ARCHIVE_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR}/${CMAKE_CFG_INTDIR})
        set(NGRAPH_ARCHIVE_INSTALL_SRC_DIRECTORY ${NGRAPH_BUILD_DIR}/\${CMAKE_INSTALL_CONFIG_NAME})
        set(CMAKE_PDB_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NGRAPH_BUILD_DIR})
    else()
        set(NGRAPH_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        set(NGRAPH_LIBRARY_INSTALL_SRC_DIRECTORY ${NGRAPH_BUILD_DIR})
    endif()
endif()

# Build destination directory for nGraph binaries and tools.
set(NGRAPH_BUILD_BIN ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    if (MSVC)
        # When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
        # These two options can't be mixed, because they requires link two imcompatiable runtime.
        set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
        set(protobuf_WITH_ZLIB  OFF CACHE BOOL "" FORCE)
        set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
    endif()
    if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
        include(cmake/external_protobuf.cmake)
    else()
        find_package(Protobuf 2.6.1 REQUIRED)
    endif()
    include(cmake/external_onnx.cmake)
    if (TARGET ext_protobuf)
        add_dependencies(onnx ext_protobuf)
    endif()
endif()

if(NGRAPH_JSON_ENABLE)
    include(cmake/external_json.cmake)
endif()

add_subdirectory(src)

if (NGRAPH_TEST_UTIL_ENABLE)
    include(cmake/external_gtest.cmake)
endif()

add_subdirectory(test)

if (NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(python)
endif()

if (NGRAPH_EXPORT_TARGETS_ENABLE)
    include(CMakePackageConfigHelpers)

    export(TARGETS ngraph NAMESPACE ngraph:: FILE "${CMAKE_CURRENT_BINARY_DIR}/ngraphTargets.cmake")

    install(EXPORT ngraphTargets
        FILE ngraphTargets.cmake
        NAMESPACE ngraph::
        DESTINATION ${NGRAPH_COMPONENT_PREFIX}cmake
        COMPONENT ngraph)

    configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/share/ngraphConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfig.cmake
        INSTALL_DESTINATION cmake)

    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ngraphConfigVersion.cmake
        VERSION ${NGRAPH_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfigVersion.cmake
        DESTINATION ${NGRAPH_COMPONENT_PREFIX}cmake
        COMPONENT ngraph)
endif()

install(DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/licenses
    DESTINATION "${NGRAPH_COMPONENT_PREFIX}."
    COMPONENT ngraph
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION "${NGRAPH_COMPONENT_PREFIX}." COMPONENT ngraph)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VERSION DESTINATION "${NGRAPH_COMPONENT_PREFIX}." COMPONENT ngraph)
