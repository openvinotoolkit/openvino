# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(NGRAPH_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

set(FRONTEND_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend/frontend_manager/include
)

# Will be used by frontends to construct frontend-specific source location paths
set(FRONTEND_BASE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/frontend
)

project (ngraph)

option(NGRAPH_UNIT_TEST_ENABLE "Control the building of unit tests" ON)
option(NGRAPH_UNIT_TEST_BACKENDS_ENABLE "Control the building of unit tests using backends" ON)
option(NGRAPH_DEBUG_ENABLE "Enable output for NGRAPH_DEBUG statements" OFF)
option(NGRAPH_ONNX_IMPORT_ENABLE "Enable ONNX importer" OFF)
option(NGRAPH_ONNX_EDITOR_ENABLE "Enable ONNX Editor" OFF)
option(NGRAPH_PDPD_FRONTEND_ENABLE "Enable PaddlePaddle FrontEnd" OFF)
option(NGRAPH_PYTHON_BUILD_ENABLE "Enable build nGraph python package wheel" OFF)
option(NGRAPH_USE_PROTOBUF_LITE "Compiles and links with protobuf-lite" OFF)

if (NGRAPH_ONNX_IMPORT_ENABLE OR NGRAPH_PDPD_FRONTEND_ENABLE)
    option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object" OFF)
endif()
if(NGRAPH_ONNX_EDITOR_ENABLE AND NOT NGRAPH_ONNX_IMPORT_ENABLE)
    message(FATAL_ERROR "ONNX Editor component requires ONNX Importer. Set NGRAPH_ONNX_IMPORT_ENABLE=ON.")
endif()

message(STATUS "NGRAPH_DEBUG_ENABLE:                  ${NGRAPH_DEBUG_ENABLE}")
message(STATUS "NGRAPH_ONNX_IMPORT_ENABLE:            ${NGRAPH_ONNX_IMPORT_ENABLE}")
message(STATUS "NGRAPH_ONNX_EDITOR_ENABLE:            ${NGRAPH_ONNX_EDITOR_ENABLE}")
message(STATUS "NGRAPH_PDPD_FRONTEND_ENABLE:          ${NGRAPH_PDPD_FRONTEND_ENABLE}")
message(STATUS "NGRAPH_PYTHON_BUILD_ENABLE:           ${NGRAPH_PYTHON_BUILD_ENABLE}")
message(STATUS "NGRAPH_USE_PROTOBUF_LITE:             ${NGRAPH_USE_PROTOBUF_LITE}")
message(STATUS "NGRAPH_UNIT_TEST_ENABLE:              ${NGRAPH_UNIT_TEST_ENABLE}")
message(STATUS "NGRAPH_UNIT_TEST_BACKENDS_ENABLE:     ${NGRAPH_UNIT_TEST_BACKENDS_ENABLE}")

# Setup CMAKE_ARGS to be forwarded to External Projects
set(NGRAPH_FORWARD_CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=${CMAKE_EXPORT_COMPILE_COMMANDS}
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=${CMAKE_POSITION_INDEPENDENT_CODE}
    -DCMAKE_DEBUG_POSTFIX=${CMAKE_DEBUG_POSTFIX}
    -DCMAKE_RELEASE_POSTFIX=${CMAKE_RELEASE_POSTFIX}
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
endif()

# Destinations
set(NGRAPH_INSTALL_LIB "deployment_tools/ngraph/${CMAKE_INSTALL_LIBDIR}")
set(NGRAPH_INSTALL_INCLUDE "deployment_tools/ngraph/${CMAKE_INSTALL_INCLUDEDIR}")
set(NGRAPH_INSTALL_BIN "deployment_tools/ngraph/${CMAKE_INSTALL_BINDIR}")

#-----------------------------------------------------------------------------------------------
# Compile Flags for nGraph...
#-----------------------------------------------------------------------------------------------

if (WIN32)
    string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
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

add_subdirectory(core)

#
# Export targets
#

set(NGRAPH_TARGETS_FILE "${CMAKE_CURRENT_BINARY_DIR}/ngraphTargets.cmake")
export(TARGETS ngraph NAMESPACE ngraph:: FILE "${NGRAPH_TARGETS_FILE}")

if(BUILD_SHARED_LIBS)
    install(EXPORT ngraphTargets
        FILE ngraphTargets.cmake
        NAMESPACE ngraph::
        DESTINATION "deployment_tools/ngraph/cmake"
        COMPONENT ngraph_dev)
endif()

configure_package_config_file(${OpenVINO_SOURCE_DIR}/cmake/templates/ngraphConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfig.cmake
    INSTALL_DESTINATION cmake)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ngraphConfigVersion.cmake
    VERSION ${IE_VERSION_MAJOR}.${IE_VERSION_MINOR}.${IE_VERSION_PATCH}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ngraphConfigVersion.cmake
    DESTINATION "deployment_tools/ngraph/cmake"
    COMPONENT ngraph_dev)

set(USE_STATIC_PROTOBUF OFF)
if (NGRAPH_PDPD_FRONTEND_ENABLE) # add more frontends here which depend on static protobuf
    set(USE_STATIC_PROTOBUF ON)
endif()

if (NGRAPH_ONNX_IMPORT_ENABLE OR USE_STATIC_PROTOBUF)
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
    set(BUILD_SHARED_LIBS ON)
    set(BUILD_STANDALONE_STATIC OFF)

    if (NOT NGRAPH_USE_SYSTEM_PROTOBUF)
        add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/protobuf ${CMAKE_BINARY_DIR}/_deps/protobuf)
    else()
        find_package(Protobuf REQUIRED)
    endif()

    if (NGRAPH_ONNX_IMPORT_ENABLE)
        # target onnx_proto will be shared lib, onnx static
        add_subdirectory(${CMAKE_SOURCE_DIR}/thirdparty/onnx ${CMAKE_BINARY_DIR}/_deps/onnx)
        if (TARGET ext_protobuf)
            add_dependencies(onnx ext_protobuf)
        endif()
    endif()

    set(BUILD_SHARED_LIBS ${BEFORE_ONNX_BUILD_SHARED_LIBS})
    unset(BEFORE_ONNX_BUILD_SHARED_LIBS)
    unset(BUILD_STANDALONE_STATIC)
endif()

add_subdirectory(frontend)

add_subdirectory(test)

if (NGRAPH_PYTHON_BUILD_ENABLE)
    add_subdirectory(python)
endif()
