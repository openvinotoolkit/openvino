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

add_definitions(-DIN_NGRAPH_LIBRARY)

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

set(NGRAPH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/ngraph CACHE INTERNAL "")

add_subdirectory(builder)
add_subdirectory(reference)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

configure_file(include/ngraph/version.in.hpp include/ngraph/version.hpp)

# Create shared library
add_library(ngraph SHARED ${LIBRARY_SRC} ${PUBLIC_HEADERS})

if(COMMAND ie_faster_build)
    ie_faster_build(ngraph
        PCH PRIVATE "src/precomp.hpp"
    )
endif()

set_target_properties(ngraph PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      C_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)

target_link_libraries(ngraph PRIVATE openvino::itt ngraph::builder ngraph::reference)

find_package(Graphviz QUIET)
if (GRAPHVIZ_FOUND)
    set_property(SOURCE pass/visualize_tree.cpp APPEND PROPERTY COMPILE_DEFINITIONS GRAPHVIZ_FOUND)
endif()

if(NGRAPH_ADDRESS_SANITIZER)
    message(STATUS "Enable Address Sanitizer")
    add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
endif()

if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(ngraph PROPERTIES
        VERSION ${NGRAPH_VERSION}
        SOVERSION ${NGRAPH_API_VERSION})
endif()
target_compile_definitions(ngraph PUBLIC NGRAPH_VERSION="${NGRAPH_VERSION}")

if (LINUX)
    # nGraph links against one or more libraries (ex. LLVM) but we don't want to
    # export these symbols as part of the DSO. This is a GNU ld (and derivatives) specific
    # option so making this portable is still an open issue. As a note for the future,
    # this is not an issue on Windows and LLVM's lld does support --exclude-libs.
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--exclude-libs,ALL")

    # GCC invokes the linker with --as-needed by default which doesn't work for us
    # because generated code needs to find symbols in these DSOs at runtime.
    # The fix below is temporary and will be removed once we find a better way
    # to do this because certain dependencies like the OpenMP runtime libraries
    # _do_ need to be linked with --as-needed with a higher priority for the
    # Intel OpenMP runtime so we don't mix libgomp and libiomp5
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,--no-as-needed")
    endif()
elseif(APPLE)
    set_property(TARGET ngraph APPEND_STRING PROPERTY LINK_FLAGS " -Wl,-rpath,@loader_path")
endif()

# Defines macro in C++ to load backend plugin
target_include_directories(ngraph PUBLIC $<BUILD_INTERFACE:${NGRAPH_INCLUDE_PATH}> $<INSTALL_INTERFACE:${NGRAPH_INSTALL_INCLUDE}>)
target_include_directories(ngraph PRIVATE ${NGRAPH_INCLUDE_DIR}
                                          ${NGRAPH_INCLUDE_DIR}/op
                                          ${NGRAPH_INCLUDE_DIR}/op/util
                                          ${NGRAPH_INCLUDE_DIR}/pass
                                          ${NGRAPH_INCLUDE_DIR}/pattern
                                          ${NGRAPH_INCLUDE_DIR}/pattern/op
                                          ${NGRAPH_INCLUDE_DIR}/runtime
                                          ${CMAKE_CURRENT_SOURCE_DIR}/src
                                          )

#Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(ngraph::ngraph ALIAS ngraph)

if (NOT WIN32)
    target_link_libraries(ngraph PRIVATE dl)
endif()

# Build subdirectories for all build types on Windows
if(WIN32)
    foreach(BUILD_TYPE Release Debug RelWithDebInfo MinSizeRel)
        if(NOT EXISTS ${NGRAPH_BUILD_DIR}/${BUILD_TYPE})
            file(MAKE_DIRECTORY ${NGRAPH_BUILD_DIR}/${BUILD_TYPE})
        endif()
    endforeach()
endif()

#-----------------------------------------------------------------------------------------------
# Installation logic...
#-----------------------------------------------------------------------------------------------

# nGraph
install(TARGETS ngraph EXPORT ngraphTargets
        RUNTIME DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        ARCHIVE DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph
        LIBRARY DESTINATION ${NGRAPH_INSTALL_LIB} COMPONENT ngraph)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${NGRAPH_INSTALL_INCLUDE}/
    COMPONENT ngraph
    FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/ngraph/version.hpp
    DESTINATION ${NGRAPH_INSTALL_INCLUDE}/ngraph
    COMPONENT ngraph)

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "nGraph library")
set(CPACK_PACKAGE_NAME "nGraph")
set(CPACK_PACKAGE_VENDOR "Intel")

set(CPACK_PACKAGE_VERSION_MAJOR ${NGRAPH_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${NGRAPH_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${NGRAPH_VERSION_PATCH})
include(CPack)
