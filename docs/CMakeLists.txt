# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_DOCKER)
    add_subdirectory(snippets)

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

    if (NGRAPH_ONNX_IMPORT_ENABLE)
        add_subdirectory(onnx_custom_op)
    endif()
    add_subdirectory(template_extension)

    set(all_docs_targets
        ie_docs_snippets
        template_extension
        templatePlugin TemplateBehaviorTests TemplateFunctionalTests)
    foreach(target_name IN LISTS all_docs_targets)
        if (TARGET ${target_name})
            set_target_properties(${target_name} PROPERTIES FOLDER docs)
            if(WIN32)
                set_target_properties(${target_name} PROPERTIES COMPILE_PDB_NAME ${target_name})
            endif()
        endif()
    endforeach()
endif()

function(build_docs)
    find_package(Doxygen REQUIRED dot)
    find_package(Python3 COMPONENTS Interpreter)
    find_package(LATEX)

    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is required to build the documentation")
    endif()

    if(NOT Python3_FOUND)
        message(FATAL_ERROR "Python3 is required to build the documentation")
    endif()

    if(NOT LATEX_FOUND)
        message(FATAL_ERROR "LATEX is required to build the documentation")
    endif()

    set(DOCS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(DOXYGEN_DIR "${OpenVINO_MAIN_SOURCE_DIR}/docs/doxygen")
    set(IE_SOURCE_DIR "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine")
    set(PYTHON_API_IN "${IE_SOURCE_DIR}/ie_bridges/python/src/openvino/inference_engine/ie_api.pyx")
    set(PYTHON_API_OUT "${DOCS_BUILD_DIR}/python_api/ie_api.pyx")
    set(C_API "${IE_SOURCE_DIR}/ie_bridges/c/include")
    set(PLUGIN_API_DIR "${DOCS_BUILD_DIR}/IE_PLUGIN_DG")
    set(NGRAPH_DIR "${OpenVINO_MAIN_SOURCE_DIR}/ngraph")
    set(NGRAPH_PY_DIR "${NGRAPH_DIR}/python/src/ngraph/")
    set(NGRAPH_CPP_DIR "${NGRAPH_DIR}/core/include/ ${NGRAPH_DIR}/frontend/onnx_import/include")

    # Preprocessing scripts
    set(DOXY_MD_FILTER "${DOXYGEN_DIR}/doxy_md_filter.py")
    set(PYX_FILTER "${DOXYGEN_DIR}/pyx_filter.py")

    file(GLOB_RECURSE doc_source_files
        LIST_DIRECTORIES true RELATIVE ${OpenVINO_MAIN_SOURCE_DIR}
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.md"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.png"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.gif"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.jpg"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.md"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.png"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.gif"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.jpg")

    configure_file(${PYTHON_API_IN} ${PYTHON_API_OUT} @ONLY)

    set(NGRAPH_CPP_CONFIG_SOURCE "${DOXYGEN_DIR}/ngraph_cpp_api.config")
    set(NGRAPH_PY_CONFIG_SOURCE "${DOXYGEN_DIR}/ngraph_py_api.config")
    set(IE_CONFIG_SOURCE "${DOXYGEN_DIR}/ie_docs.config")
    set(C_CONFIG_SOURCE "${DOXYGEN_DIR}/ie_c_api.config")
    set(PY_CONFIG_SOURCE "${DOXYGEN_DIR}/ie_py_api.config")
    set(PLUGIN_CONFIG_SOURCE "${DOXYGEN_DIR}/ie_plugin_api.config")

    set(NGRAPH_CPP_CONFIG_BUILD "${DOCS_BUILD_DIR}/ngraph_cpp_api.config")
    set(NGRAPH_PY_CONFIG_BUILD "${DOCS_BUILD_DIR}/ngraph_py_api.config")
    set(IE_CONFIG_BUILD "${DOCS_BUILD_DIR}/ie_docs.config")
    set(C_CONFIG_BUILD "${DOCS_BUILD_DIR}/ie_c_api.config")
    set(PY_CONFIG_BUILD "${DOCS_BUILD_DIR}/ie_py_api.config")
    set(PLUGIN_CONFIG_BUILD "${DOCS_BUILD_DIR}/ie_plugin_api.config")

    set(NGRAPH_CPP_LAYOUT_SOURCE "${DOXYGEN_DIR}/ngraph_cpp_api.xml")
    set(NGRAPH_PY_LAYOUT_SOURCE "${DOXYGEN_DIR}/ngraph_py_api.xml")
    set(IE_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_docs.xml")
    set(C_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_c_api.xml")
    set(PY_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_py_api.xml")
    set(PLUGIN_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_plugin_api.xml")

    set(NGRAPH_CPP_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ngraph_cpp_api.xml")
    set(NGRAPH_PY_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ngraph_py_api.xml")
    set(IE_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_docs.xml")
    set(C_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_c_api.xml")
    set(PY_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_py_api.xml")
    set(PLUGIN_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_plugin_api.xml")

    # Tables of contents
    configure_file(${NGRAPH_CPP_LAYOUT_SOURCE} ${NGRAPH_CPP_LAYOUT_BUILD} @ONLY)
    configure_file(${NGRAPH_PY_LAYOUT_SOURCE} ${NGRAPG_PY_LAYOUT_BUILD} @ONLY)
    configure_file(${IE_LAYOUT_SOURCE} ${IE_LAYOUT_BUILD} @ONLY)
    configure_file(${C_LAYOUT_SOURCE} ${C_LAYOUT_BUILD} @ONLY)
    configure_file(${PY_LAYOUT_SOURCE} ${PY_LAYOUT_BUILD} @ONLY)
    configure_file(${PLUGIN_LAYOUT_SOURCE} ${PLUGIN_LAYOUT_BUILD} @ONLY)

    # Doxygen config files
    configure_file(${NGRAPH_CPP_CONFIG_SOURCE} ${NGRAPH_CPP_CONFIG_BUILD} @ONLY)
    configure_file(${NGRAPH_PY_CONFIG_SOURCE} ${NGRAPH_PY_CONFIG_BUILD} @ONLY)
    configure_file(${IE_CONFIG_SOURCE} ${IE_CONFIG_BUILD} @ONLY)
    configure_file(${C_CONFIG_SOURCE} ${C_CONFIG_BUILD} @ONLY)
    configure_file(${PY_CONFIG_SOURCE} ${PY_CONFIG_BUILD} @ONLY)
    configure_file(${PLUGIN_CONFIG_SOURCE} ${PLUGIN_CONFIG_BUILD} @ONLY)

    # Preprocessing scripts
    set(DOXY_MD_FILTER "${DOXYGEN_DIR}/doxy_md_filter.py")
    set(PYX_FILTER "${DOXYGEN_DIR}/pyx_filter.py")

    # nGraph C++ API

    add_custom_target(ngraph_cpp_api
                      COMMAND ${DOXYGEN_EXECUTABLE} ${NGRAPH_CPP_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    # nGraph Python API

    add_custom_target(ngraph_py_api
                      COMMAND ${DOXYGEN_EXECUTABLE} ${NGRAPH_PY_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    # C API

    add_custom_target(c_api
                      COMMAND ${DOXYGEN_EXECUTABLE} ${C_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generating C API Reference"
                      VERBATIM)

    # Python API

    add_custom_target(py_api
                      COMMAND ${DOXYGEN_EXECUTABLE} ${PY_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generating Python API Reference"
                      VERBATIM)

    add_custom_command(TARGET py_api
                       PRE_BUILD
                       COMMAND ${Python3_EXECUTABLE} ${PYX_FILTER} ${PYTHON_API_OUT}
                       COMMENT "Pre-process Python API")

    # Preprocess docs

    add_custom_target(preprocess_docs
                      COMMENT "Pre-process docs"
                      VERBATIM)

    foreach(source_file ${doc_source_files})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy
            "${OpenVINO_MAIN_SOURCE_DIR}/${source_file}" "${DOCS_BUILD_DIR}/${source_file}")
    endforeach()

    add_custom_command(TARGET preprocess_docs
                       PRE_BUILD
                       ${commands}
                       COMMAND ${Python3_EXECUTABLE} ${DOXY_MD_FILTER} ${DOCS_BUILD_DIR}
                       COMMENT "Pre-process markdown and image links")

    # IE dev guide and C++ API

    add_custom_target(ie_docs
                      DEPENDS ngraph_cpp_api preprocess_docs
                      COMMAND ${DOXYGEN_EXECUTABLE} ${IE_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    # Plugin API

    add_custom_target(plugin_api
                      DEPENDS ie_docs
                      COMMAND ${DOXYGEN_EXECUTABLE} ${PLUGIN_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generating Plugin API Reference"
                      VERBATIM)

    # Umbrella OpenVINO target

    add_custom_target(openvino_docs
                      DEPENDS ngraph_cpp_api ngraph_py_api c_api py_api ie_docs plugin_api
                      COMMENT "Generating OpenVINO documentation"
                      VERBATIM)

    set_target_properties(openvino_docs ie_docs c_api py_api preprocess_docs plugin_api
                          PROPERTIES FOLDER docs)

    find_program(browser NAMES xdg-open)
    if(browser)
        add_custom_target(ie_docs_open
                          COMMAND ${browser} "${OpenVINO_MAIN_SOURCE_DIR}/docs/html/index.html"
                          DEPENDS ie_docs
                          COMMENT "Open OpenVINO documentation"
                          VERBATIM)
        set_target_properties(ie_docs_open PROPERTIES FOLDER docs)
    endif()
endfunction()

if(ENABLE_DOCS)
    build_docs()
endif()
