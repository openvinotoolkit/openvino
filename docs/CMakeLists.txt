# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(examples)

# Detect nGraph
find_package(ngraph QUIET)
if(NOT ngraph_FOUND)
    set(ngraph_DIR ${CMAKE_BINARY_DIR}/ngraph)
endif()

# Detect InferenceEngine
find_package(InferenceEngine QUIET)
if(NOT InferenceEngine_FOUND)
    set(InferenceEngine_DIR ${CMAKE_BINARY_DIR})
endif()

add_subdirectory(template_extension)

set(all_docs_targets
    ie_docs_examples
    template_extension
    templatePlugin TemplateBehaviorTests TemplateFunctionalTests)
foreach(target_name IN LISTS all_docs_targets)
    if (TARGET ${target_name})
        set_target_properties(${target_name} PROPERTIES FOLDER docs)
    endif()
endforeach()

# OpenVINO docs

set(OPENVINO_DOCS_PATH "" CACHE PATH "Path to openvino-documentation local repository")
set(args "")

if(OPENVINO_DOCS_PATH)
    set(args "${args} ovinodoc_path:${OPENVINO_DOCS_PATH}")
endif()

file(GLOB_RECURSE docs_files "${OpenVINO_MAIN_SOURCE_DIR}/docs")
file(GLOB_RECURSE include_files "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/include")
file(GLOB_RECURSE ovino_files "${OPENVINO_DOCS_PATH}")

add_custom_target(ie_docs
                  COMMAND ./build_docs.sh ${args}
                  WORKING_DIRECTORY "${OpenVINO_MAIN_SOURCE_DIR}/docs/build_documentation"
                  COMMENT "Generating OpenVINO documentation"
                  SOURCES ${docs_files} ${include_files} ${ovino_files}
                  VERBATIM)
set_target_properties(ie_docs PROPERTIES FOLDER docs)

find_program(browser NAMES xdg-open)
if(browser)
    add_custom_target(ie_docs_open
                      COMMAND ${browser} "${OpenVINO_MAIN_SOURCE_DIR}/doc/html/index.html"
                      DEPENDS ie_docs
                      COMMENT "Open OpenVINO documentation"
                      VERBATIM)
    set_target_properties(ie_docs_open PROPERTIES FOLDER docs)
endif()
