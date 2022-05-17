#===============================================================================
# Copyright 2016-2021 Intel Corporation
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
#===============================================================================

# Locates Doxygen and configures documentation generation
#===============================================================================

if(Doxygen_cmake_included)
    return()
endif()
set(Doxygen_cmake_included true)

if(NOT DNNL_IS_MAIN_PROJECT)
    return()
endif()

find_package(Doxygen)
if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/reference)
    set(DOXYGEN_STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/doxygen.stamp)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY)
    file(GLOB_RECURSE HEADERS
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/*.h
        ${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/*.hpp
        )
    file(GLOB_RECURSE DOX
        ${PROJECT_SOURCE_DIR}/doc/*
        )
    file(GLOB_RECURSE EXAMPLES
        ${PROJECT_SOURCE_DIR}/examples/*
        )
    add_custom_command(
        OUTPUT ${DOXYGEN_STAMP_FILE}
        DEPENDS ${HEADERS} ${DOX} ${EXAMPLES}
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        COMMAND ${CMAKE_COMMAND} -E touch ${DOXYGEN_STAMP_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation in .xml format with Doxygen" VERBATIM)
    add_custom_target(doc_doxygen DEPENDS ${DOXYGEN_STAMP_FILE})

    if(NOT DNNL_INSTALL_MODE STREQUAL "BUNDLE")
        install(
            DIRECTORY ${DOXYGEN_OUTPUT_DIR}
            DESTINATION share/doc/${DNNL_LIBRARY_NAME} OPTIONAL)
    endif()
endif(DOXYGEN_FOUND)
