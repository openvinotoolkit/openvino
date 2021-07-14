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
option(NGRAPH_USE_PROTOBUF_LITE "Compiles and links with protobuf-lite" OFF)

if (NGRAPH_ONNX_IMPORT_ENABLE OR NGRAPH_PDPD_FRONTEND_ENABLE)
    option(NGRAPH_USE_SYSTEM_PROTOBUF "Use system provided Protobuf shared object" OFF)
endif()

if(NGRAPH_ONNX_EDITOR_ENABLE AND NOT NGRAPH_ONNX_IMPORT_ENABLE)
    message(FATAL_ERROR "ONNX Editor component requires ONNX Importer. Set NGRAPH_ONNX_IMPORT_ENABLE=ON.")
endif()

message(STATUS "NGRAPH_DEBUG_ENABLE:                  ${NGRAPH_DEBUG_ENABLE}")
message(STATUS "NGRAPH_UNIT_TEST_ENABLE:              ${NGRAPH_UNIT_TEST_ENABLE}")
message(STATUS "NGRAPH_UNIT_TEST_BACKENDS_ENABLE:     ${NGRAPH_UNIT_TEST_BACKENDS_ENABLE}")

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

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

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
export(TARGETS ngraph NAMESPACE ngraph:: APPEND FILE "${NGRAPH_TARGETS_FILE}")

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

add_subdirectory(frontend)

add_subdirectory(test)

if (ENABLE_PYTHON)
    add_subdirectory(python)
endif()
