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

include(cmake/Modules/git_tags.cmake)

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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/VERSION.in VERSION)

message(STATUS "NGRAPH_VERSION ${NGRAPH_VERSION}")
message(STATUS "NGRAPH_VERSION_SHORT ${NGRAPH_VERSION_SHORT}")
message(STATUS "NGRAPH_WHEEL_VERSION ${NGRAPH_WHEEL_VERSION}")
message(STATUS "NGRAPH_API_VERSION ${NGRAPH_API_VERSION}")

set(NGRAPH_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

if (APPLE)
    # Enable MACOS_RPATH by default.
    cmake_policy(SET CMP0042 NEW)
endif()

project (ngraph)

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

option(NGRAPH_UNIT_TEST_ENABLE "Control the building of unit tests" ON)
option(NGRAPH_TEST_UTIL_ENABLE "Control the building of test utility" ON)
option(NGRAPH_INTERPRETER_ENABLE "Control the building of the INTERPRETER backend" ON)
option(NGRAPH_DEBUG_ENABLE "Enable output for NGRAPH_DEBUG statements" OFF)
option(NGRAPH_ONNX_IMPORT_ENABLE "Enable ONNX importer" OFF)
option(NGRAPH_LIB_VERSIONING_ENABLE "Enable shared library versioning" OFF)
option(NGRAPH_PYTHON_BUILD_ENABLE "Enable build nGraph python package wheel" OFF)
option(NGRAPH_DYNAMIC_COMPONENTS_ENABLE "Enable dynamic loading of components" ON)
option(NGRAPH_EXPORT_TARGETS_ENABLE "Enable exporting nGraph cmake export targets" ON)
option(NGRAPH_WARNINGS_AS_ERRORS "Make all nGraph compile-time warnings into errors" OFF)
option(NGRAPH_ADDRESS_SANITIZER_ENABLE "Compiles and links with Address Sanitizer" OFF)
option(NGRAPH_THREAD_SANITIZER_ENABLE "Compiles and links with Thread Sanitizer" OFF)
option(NGRAPH_UB_SANITIZER_ENABLE "Compiles and links with Undefined Behavior Sanitizer" OFF)
option(NGRAPH_USE_PROTOBUF_LITE "Compiles and links with protobuf-lite" OFF)

if (NGRAPH_ONNX_IMPORT_ENABLE)
    option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object" OFF)
endif()

if (NGRAPH_UNIT_TEST_ENABLE)
    if (NOT NGRAPH_TEST_UTIL_ENABLE)
        message(WARNING "Unit test needs test util: Turning NGRAPH_TEST_UTIL_ENABLE ON!")
        set(NGRAPH_TEST_UTIL_ENABLE ON)
    endif()
endif()

message(STATUS "NGRAPH_ADDRESS_SANITIZER_ENABLE:      ${NGRAPH_ADDRESS_SANITIZER_ENABLE}")
message(STATUS "NGRAPH_DEBUG_ENABLE:                  ${NGRAPH_DEBUG_ENABLE}")
message(STATUS "NGRAPH_DYNAMIC_COMPONENTS_ENABLE:     ${NGRAPH_DYNAMIC_COMPONENTS_ENABLE}")
message(STATUS "NGRAPH_EXPORT_TARGETS_ENABLE:         ${NGRAPH_EXPORT_TARGETS_ENABLE}")
message(STATUS "NGRAPH_INTERPRETER_ENABLE:            ${NGRAPH_INTERPRETER_ENABLE}")
message(STATUS "NGRAPH_LIB_VERSIONING_ENABLE:         ${NGRAPH_LIB_VERSIONING_ENABLE}")
message(STATUS "NGRAPH_ONNX_IMPORT_ENABLE:            ${NGRAPH_ONNX_IMPORT_ENABLE}")
message(STATUS "NGRAPH_PYTHON_BUILD_ENABLE:           ${NGRAPH_PYTHON_BUILD_ENABLE}")
message(STATUS "NGRAPH_TEST_UTIL_ENABLE:              ${NGRAPH_TEST_UTIL_ENABLE}")
message(STATUS "NGRAPH_THREAD_SANITIZER_ENABLE:       ${NGRAPH_THREAD_SANITIZER_ENABLE}")
message(STATUS "NGRAPH_TOOLS_ENABLE:                  ${NGRAPH_TOOLS_ENABLE}")
message(STATUS "NGRAPH_UB_SANITIZER_ENABLE:           ${NGRAPH_UB_SANITIZER_ENABLE}")
message(STATUS "NGRAPH_USE_PROTOBUF_LITE:             ${NGRAPH_USE_PROTOBUF_LITE}")
message(STATUS "NGRAPH_UNIT_TEST_ENABLE:              ${NGRAPH_UNIT_TEST_ENABLE}")
message(STATUS "NGRAPH_WARNINGS_AS_ERRORS:            ${NGRAPH_WARNINGS_AS_ERRORS}")

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
    set(CMAKE_INSTALL_BINDIR "bin" CACHE STRING "User executables (bin)")
    set(CMAKE_INSTALL_LIBDIR "lib" CACHE STRING "Object code libraries (lib)")
    set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "C header files (include)")
    set(CMAKE_INSTALL_DOCDIR "doc" CACHE STRING "Document files (doc)")
    mark_as_advanced(CMAKE_INSTALL_BINDIR CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR, CMAKE_INSTALL_DOCDIR)
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

if (WIN32)
    string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if (NOT WIN32 AND NGRAPH_WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_CRT_SECURE_NO_WARNINGS")
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
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ADDRESS_SANITIZER_FLAGS}")
endif()

if(NGRAPH_THREAD_SANITIZER_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")
endif()

if(NGRAPH_UB_SANITIZER_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=float-divide-by-zero -fsanitize=undefined -fsanitize=float-cast-overflow -fPIE -pie")
endif()

if(NGRAPH_ADDRESS_SANITIZER_ENABLE OR NGRAPH_THREAD_SANITIZER_ENABLE OR NGRAPH_UB_SANITIZER_ENABLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -g -fno-omit-frame-pointer")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -g -fno-omit-frame-pointer")
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

#-----------------------------------------------------------------------------------------------
# enable or disable output from NGRAPH_DEBUG statements
#-----------------------------------------------------------------------------------------------
if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
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

# Build destination directory for nGraph binaries and tools.

if(NOT DEFINED EXTERNAL_PROJECTS_ROOT)
    set(EXTERNAL_PROJECTS_ROOT ${CMAKE_CURRENT_BINARY_DIR})
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE)
    if (MSVC)
        # When we build dll libraries. These flags make sure onnx and protobuf build with /MD, not /MT.
        # These two options can't be mixed, because they requires link two imcompatiable runtime.
        set(protobuf_WITH_ZLIB  OFF CACHE BOOL "" FORCE)

        if(NOT DEFINED ONNX_USE_MSVC_STATIC_RUNTIME)
            set(ONNX_USE_MSVC_STATIC_RUNTIME OFF)
        endif()
        if(NOT DEFINED protobuf_MSVC_STATIC_RUNTIME)
            set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "Link protobuf to static runtime libraries" FORCE)
        endif()
    endif()

    set(BEFORE_ONNX_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
        include(cmake/external_protobuf.cmake)
    else()
        find_package(Protobuf 2.6.1 REQUIRED)
    endif()
    include(cmake/external_onnx.cmake)
    if (TARGET ext_protobuf)
        add_dependencies(onnx ext_protobuf)
    endif()
    set(BUILD_SHARED_LIBS ${BEFORE_ONNX_BUILD_SHARED_LIBS})
    unset(BEFORE_ONNX_BUILD_SHARED_LIBS)
endif()

add_subdirectory(core)

if (NGRAPH_EXPORT_TARGETS_ENABLE)
    include(CMakePackageConfigHelpers)
    set(NGRAPH_TARGETS_FILE "${CMAKE_CURRENT_BINARY_DIR}/ngraphTargets.cmake")
    export(TARGETS ngraph NAMESPACE ngraph:: FILE "${NGRAPH_TARGETS_FILE}")

    if(BUILD_SHARED_LIBS)
        install(EXPORT ngraphTargets
            FILE ngraphTargets.cmake
            NAMESPACE ngraph::
            DESTINATION ${NGRAPH_COMPONENT_PREFIX}cmake
            COMPONENT ngraph)
    endif()

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

add_subdirectory(frontend)

if (NGRAPH_TEST_UTIL_ENABLE)
    set(BEFORE_GTEST_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    include(cmake/external_gtest.cmake)
    set(BUILD_SHARED_LIBS ${BEFORE_GTEST_BUILD_SHARED_LIBS})
    unset(BEFORE_GTEST_BUILD_SHARED_LIBS)
endif()

add_subdirectory(test)

if (NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(python)
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/VERSION DESTINATION "${NGRAPH_COMPONENT_PREFIX}." COMPONENT ngraph)
