# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE LIBRARY_PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(FRONTEND_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})
source_group("public include" FILES ${LIBRARY_PUBLIC_HEADERS})

# Create shared library
add_library(frontend_manager SHARED ${LIBRARY_SRC} ${LIBRARY_HEADERS} ${LIBRARY_PUBLIC_HEADERS})
add_library(ngraph::frontend_manager ALIAS frontend_manager)

target_link_libraries(frontend_manager PRIVATE ${CMAKE_DL_LIBS} ngraph)

add_clang_format_target(frontend_manager_clang FOR_TARGETS frontend_manager)

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME frontend_manager
                           FILEDESCRIPTION "Manager of OpenVINO nGraph Frontends")
endif()

set(FRONTEND_INSTALL_INCLUDE "${NGRAPH_INSTALL_INCLUDE}/ngraph/frontend/generic")
target_include_directories(frontend_manager PUBLIC $<BUILD_INTERFACE:${FRONTEND_INCLUDE_DIR}>
                                                       $<INSTALL_INTERFACE:${FRONTEND_INSTALL_INCLUDE}>)
target_include_directories(frontend_manager PRIVATE ${NGRAPH_INCLUDE_PATH} ${FRONTEND_INCLUDE_DIR})

target_include_directories(frontend_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
