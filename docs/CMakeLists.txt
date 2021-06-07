# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_DOCKER)
    if(CMAKE_COMPILER_IS_GNUCXX)
        ie_add_compiler_flags(-Wall)
    endif()

    add_subdirectory(snippets)

    # Detect nGraph
    find_package(ngraph QUIET
                 PATHS "${CMAKE_BINARY_DIR}/ngraph"
                 NO_DEFAULT_PATH)
    if(NOT ngraph_FOUND)
        set(ngraph_DIR ${CMAKE_BINARY_DIR}/ngraph)
    endif()

    # Detect InferenceEngine
    find_package(InferenceEngine QUIET
                 PATHS "${CMAKE_BINARY_DIR}"
                 NO_DEFAULT_PATH)
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

set(LINKCHECKER_PY "" CACHE FILEPATH "Path to linkchecker.py for documentation check")
set(OMZ_DOCS_DIR "" CACHE PATH "Path to open_model_zoo documentation")
set(WORKBENCH_DOCS_DIR "" CACHE PATH "Path to workbench documentation")
set(POT_DOCS_DIR "" CACHE PATH "Path to post-training-compression-tool documentation")
set(GST_DOCS_DIR "" CACHE PATH "Path to gst-video-analytics documentation")

function(build_docs)
    find_package(Doxygen REQUIRED dot)
    find_package(PythonInterp 3 REQUIRED)
    find_package(LATEX REQUIRED)

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -m pip show lxml
        RESULT_VARIABLE PIP_EXIT_CODE
        OUTPUT_QUIET
    )

    if (NOT ${PIP_EXIT_CODE} EQUAL 0)
        message(FATAL_ERROR "lxml package is not installed. Please use \"pip install lxml\".")
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
    set(NGRAPH_CPP_DIR "${NGRAPH_DIR}/core/include/" "${NGRAPH_DIR}/frontend/onnx_import/include")

    # Preprocessing scripts
    set(DOXY_MD_FILTER "${DOXYGEN_DIR}/doxy_md_filter.py")
    set(DOXY_LAYOUT_SCRIPT "${DOXYGEN_DIR}/build_main_layout.py")
    set(DOXY_LOG_SCRIPT "${DOXYGEN_DIR}/log.py")
    set(PYX_FILTER "${DOXYGEN_DIR}/pyx_filter.py")

    # assets dir
    set(ASSETS_DIR "${DOXYGEN_DIR}/assets")
    
    # header and footer
    set(HEADER_SOURCE "${DOXYGEN_DIR}/header.html.in")
    set(FOOTER_SOURCE "${DOXYGEN_DIR}/footer.html.in")
    set(HEADER_BUILD "${DOCS_BUILD_DIR}/header.html")
    set(FOOTER_BUILD "${DOCS_BUILD_DIR}/footer.html")

    configure_file(${HEADER_SOURCE} ${HEADER_BUILD} @ONLY)
    configure_file(${FOOTER_SOURCE} ${FOOTER_BUILD} @ONLY)

    file(GLOB_RECURSE doc_source_files
        LIST_DIRECTORIES true RELATIVE ${OpenVINO_MAIN_SOURCE_DIR}
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.md"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.png"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.gif"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.jpg"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.svg"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.md"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.png"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.gif"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.jpg"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.svg")

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
    set(OPENVINO_LAYOUT_SOURCE "${DOXYGEN_DIR}/openvino_docs.xml")
    set(C_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_c_api.xml")
    set(PY_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_py_api.xml")
    set(PLUGIN_LAYOUT_SOURCE "${DOXYGEN_DIR}/ie_plugin_api.xml")

    set(NGRAPH_CPP_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ngraph_cpp_api.xml")
    set(NGRAPH_PY_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ngraph_py_api.xml")
    set(IE_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_docs.xml")
    set(OPENVINO_LAYOUT_BUILD "${DOCS_BUILD_DIR}/openvino_docs.xml")
    set(C_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_c_api.xml")
    set(PY_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_py_api.xml")
    set(PLUGIN_LAYOUT_BUILD "${DOCS_BUILD_DIR}/ie_plugin_api.xml")

    # out dirs
    set(OUTPUT_DIRECTORY "${DOCS_BUILD_DIR}/html")
    set(IE_OUTPUT "${OUTPUT_DIRECTORY}")
    set(C_OUTPUT "${OUTPUT_DIRECTORY}/ie_c_api")
    set(PY_OUTPUT "${OUTPUT_DIRECTORY}/ie_python_api")
    set(PLUGIN_OUTPUT "${OUTPUT_DIRECTORY}/ie_plugin_api")
    set(NGRAPH_CPP_OUTPUT "${OUTPUT_DIRECTORY}/ngraph_cpp_api")
    set(NGRAPH_PY_OUTPUT "${OUTPUT_DIRECTORY}/ngraph_python_api")

    # Tables of contents
    configure_file(${NGRAPH_CPP_LAYOUT_SOURCE} ${NGRAPH_CPP_LAYOUT_BUILD} @ONLY)
    configure_file(${NGRAPH_PY_LAYOUT_SOURCE} ${NGRAPH_PY_LAYOUT_BUILD} @ONLY)
    configure_file(${IE_LAYOUT_SOURCE} ${IE_LAYOUT_BUILD} @ONLY)
    configure_file(${OPENVINO_LAYOUT_SOURCE} ${OPENVINO_LAYOUT_BUILD} @ONLY)
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
                      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSETS_DIR} ${NGRAPH_CPP_OUTPUT}/assets
                      COMMAND ${DOXYGEN_EXECUTABLE} ${NGRAPH_CPP_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    # nGraph Python API

    add_custom_target(ngraph_py_api
                      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSETS_DIR} ${NGRAPH_PY_OUTPUT}/assets
                      COMMAND ${DOXYGEN_EXECUTABLE} ${NGRAPH_PY_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    # C API

    add_custom_target(c_api
                      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSETS_DIR} ${C_OUTPUT}/assets
                      COMMAND ${DOXYGEN_EXECUTABLE} ${C_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generating C API Reference"
                      VERBATIM)

    # Python API

    add_custom_target(py_api
                      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSETS_DIR} ${PY_OUTPUT}/assets
                      COMMAND ${DOXYGEN_EXECUTABLE} ${PY_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generating Python API Reference"
                      VERBATIM)

    add_custom_command(TARGET py_api
                       PRE_BUILD
                       COMMAND ${PYTHON_EXECUTABLE} ${PYX_FILTER} ${PYTHON_API_OUT}
                       COMMENT "Pre-process Python API")

    # Preprocess docs

    add_custom_target(preprocess_docs
                      COMMENT "Pre-process docs"
                      VERBATIM)

    # ovino doc files
    file(GLOB_RECURSE ovino_doc_files
        LIST_DIRECTORIES true RELATIVE ${OpenVINO_MAIN_SOURCE_DIR}
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.md"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.png"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.gif"
        "${OpenVINO_MAIN_SOURCE_DIR}/docs/*.jpg"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.md"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.png"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.gif"
        "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine/*.jpg")

    foreach(source_file ${ovino_doc_files})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy
            "${OpenVINO_MAIN_SOURCE_DIR}/${source_file}" "${DOCS_BUILD_DIR}/openvino/${source_file}")
    endforeach()

    # omz doc files
    if(EXISTS "${OMZ_DOCS_DIR}")
        get_filename_component(OMZ_DOCS_DIR "${OMZ_DOCS_DIR}" ABSOLUTE)

        file(GLOB_RECURSE omz_doc_files
            LIST_DIRECTORIES true RELATIVE ${OMZ_DOCS_DIR}
            "${OMZ_DOCS_DIR}/*.md"
            "${OMZ_DOCS_DIR}/*.png"
            "${OMZ_DOCS_DIR}/*.gif"
            "${OMZ_DOCS_DIR}/*.jpg")

        foreach(source_file ${omz_doc_files})
            list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy
                "${OMZ_DOCS_DIR}/${source_file}" "${DOCS_BUILD_DIR}/omz/${source_file}")
        endforeach()
        configure_file("${OMZ_DOCS_DIR}/omz_docs.xml" "${DOCS_BUILD_DIR}/omz_docs.xml" @ONLY)
    endif()

    # workbench doc files
    if(EXISTS "${WORKBENCH_DOCS_DIR}")
        get_filename_component(WORKBENCH_DOCS_DIR "${WORKBENCH_DOCS_DIR}" ABSOLUTE)

        file(GLOB_RECURSE workbench_doc_files
            LIST_DIRECTORIES true RELATIVE ${WORKBENCH_DOCS_DIR}
            "${WORKBENCH_DOCS_DIR}/*.md"
            "${WORKBENCH_DOCS_DIR}/*.png"
            "${WORKBENCH_DOCS_DIR}/*.gif"
            "${WORKBENCH_DOCS_DIR}/*.jpg")

        foreach(source_file ${workbench_doc_files})
            list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy
                "${WORKBENCH_DOCS_DIR}/${source_file}" "${DOCS_BUILD_DIR}/workbench/${source_file}")
        endforeach()
        configure_file("${WORKBENCH_DOCS_DIR}/docs/Workbench_DG/workbench_docs.xml" "${DOCS_BUILD_DIR}/workbench_docs.xml" @ONLY)
    endif()

    # pot doc files
    if(EXISTS "${POT_DOCS_DIR}")
        get_filename_component(POT_DOCS_DIR "${POT_DOCS_DIR}" ABSOLUTE)

        file(GLOB_RECURSE pot_doc_files
            LIST_DIRECTORIES true RELATIVE ${POT_DOCS_DIR}
            "${POT_DOCS_DIR}/*.md"
            "${POT_DOCS_DIR}/*.png"
            "${POT_DOCS_DIR}/*.gif"
            "${POT_DOCS_DIR}/*.jpg")

        foreach(source_file ${pot_doc_files})
            list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy
                "${POT_DOCS_DIR}/${source_file}" "${DOCS_BUILD_DIR}/pot/${source_file}")
        endforeach()
        configure_file("${POT_DOCS_DIR}/docs/pot_docs.xml" "${DOCS_BUILD_DIR}/pot_docs.xml" @ONLY)
    endif()

    # gst doc files
    if(EXISTS "${GST_DOCS_DIR}")
        get_filename_component(GST_DOCS_DIR "${GST_DOCS_DIR}" ABSOLUTE)

        file(GLOB_RECURSE gst_doc_files
            LIST_DIRECTORIES true RELATIVE ${GST_DOCS_DIR}
            "${GST_DOCS_DIR}/*.md"
            "${GST_DOCS_DIR}/*.png"
            "${GST_DOCS_DIR}/*.gif"
            "${GST_DOCS_DIR}/*.jpg")

        foreach(source_file ${gst_doc_files})
            list(APPEND commands COMMAND ${CMAKE_COMMAND} -E copy
                "${GST_DOCS_DIR}/${source_file}" "${DOCS_BUILD_DIR}/gst/${source_file}")
        endforeach()
    endif()

    add_custom_command(TARGET preprocess_docs
                       PRE_BUILD
                       ${commands}
                       COMMAND ${PYTHON_EXECUTABLE} ${DOXY_LAYOUT_SCRIPT} --openvino ${OPENVINO_LAYOUT_BUILD}
                       COMMAND ${PYTHON_EXECUTABLE} ${DOXY_MD_FILTER} ${DOCS_BUILD_DIR}
                       COMMENT "Pre-process markdown and image links")

    # IE dev guide and C++ API

    add_custom_target(ie_docs
                      DEPENDS ngraph_cpp_api preprocess_docs
                      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSETS_DIR} ${IE_OUTPUT}/assets
                      COMMAND ${DOXYGEN_EXECUTABLE} ${IE_CONFIG_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    # Plugin API

    add_custom_target(plugin_api
                      DEPENDS ngraph_cpp_api ie_docs
                      COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSETS_DIR} ${PLUGIN_OUTPUT}/assets
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
                          ngraph_py_api ngraph_cpp_api
                          PROPERTIES FOLDER docs)

    add_custom_command(TARGET openvino_docs
                       POST_BUILD
                       COMMAND ${PYTHON_EXECUTABLE} ${DOXY_LOG_SCRIPT} --log "${DOCS_BUILD_DIR}/ie_docs.log"
                                                                        --include_omz $<BOOL:${OMZ_DOCS_DIR}>
                                                                        --include_wb $<BOOL:${WORKBENCH_DOCS_DIR}>
                                                                        --include_pot $<BOOL:${POT_DOCS_DIR}>
                                                                        --include_gst $<BOOL:${GST_DOCS_DIR}>
                       COMMENT "Parse doxygen log to find errors."
                       VERBATIM)

    # added linkcheker

    if(EXISTS "${LINKCHECKER_PY}")
        add_custom_target(docs_check
                            COMMAND ${PYTHON_EXECUTABLE} "${LINKCHECKER_PY}" -v "${DOCS_BUILD_DIR}/html/"
                            COMMENT "Check links in generated documentation"
                            WORKING_DIRECTORY "${DOCS_BUILD_DIR}"
                            VERBATIM)
        set_target_properties(docs_check PROPERTIES FOLDER docs)
    endif()

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
