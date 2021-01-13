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

add_subdirectory(builder)
add_subdirectory(reference)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

configure_file(include/ngraph/version.in.hpp include/ngraph/version.hpp)

# Create static or shared library depending on BUILD_SHARED_LIBS
add_library(ngraph ${LIBRARY_SRC} ${PUBLIC_HEADERS})


if(COMMAND ie_faster_build)
    ie_faster_build(ngraph
        UNITY
        PCH PRIVATE "src/precomp.hpp"
    )
endif()

if(COMMAND ie_add_api_validator_post_build_step)
    ie_add_api_validator_post_build_step(TARGET ngraph)
endif()

if(COMMAND ie_add_vs_version_file)
    ie_add_vs_version_file(NAME ngraph
                           FILEDESCRIPTION "nGraph library")
endif()

target_link_libraries(ngraph PRIVATE openvino::conditional_compilation openvino::itt ngraph::builder ngraph::reference)

if(NGRAPH_ADDRESS_SANITIZER)
    message(STATUS "Enable Address Sanitizer")
    add_compile_options(-g -fsanitize=address -fno-omit-frame-pointer)
endif()

if(NGRAPH_LIB_VERSIONING_ENABLE)
    set_target_properties(ngraph PROPERTIES
        VERSION ${NGRAPH_VERSION}
        SOVERSION ${NGRAPH_API_VERSION})
endif()

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(ngraph PUBLIC NGRAPH_STATIC_LIBRARY)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # ngraph is linked against ngraph_builders, ngraph_reference static libraries
    # which include ngraph headers with dllimport attribute. Linker complains about it
    # but no way to fix this: linking with no attribute defaults to dllexport and we have
    # multiple defitions for ngraph symbols.
    #
    # The possible way is to use object libraries for ngraph_builders, ngraph_reference
    # but it's not convinient since these libraries are exported from build tree
    # and it's better to use them as static libraries in 3rd party projects
    set_target_properties(ngraph PROPERTIES LINK_FLAGS "/IGNORE:4217,4286")
endif()

# Defines macro in C++ to load backend plugin
target_include_directories(ngraph PUBLIC $<BUILD_INTERFACE:${NGRAPH_INCLUDE_PATH}>
                                         $<INSTALL_INTERFACE:${NGRAPH_INSTALL_INCLUDE}>
                                  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                                          ${CMAKE_CURRENT_BINARY_DIR}/include)

#Add an alias so that library can be used inside the build tree, e.g. when testing
add_library(ngraph::ngraph ALIAS ngraph)

if (NOT WIN32)
    target_link_libraries(ngraph PRIVATE ${CMAKE_DL_LIBS})
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
