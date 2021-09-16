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

set(LINKCHECKER_PY "" CACHE FILEPATH "Path to linkchecker.py for documentation check dir.")
set(OMZ_DOCS_DIR "" CACHE PATH "Path to open_model_zoo documentation dir.")
set(WORKBENCH_DOCS_DIR "" CACHE PATH "Path to workbench documentation dir.")
set(POT_DOCS_DIR "" CACHE PATH "Path to post-training-compression-tool documentation dir.")
set(GST_DOCS_DIR "" CACHE PATH "Path to gst-video-analytics documentation dir.")
set(OPENVINO_NOTEBOOKS_DIR "" CACHE PATH "Path to openvino notebooks dir.")

# python packages
set(NGRAPH_PYTHON_PACKAGE "" CACHE PATH "Path to generated ngraph python package.")
set(IE_PYTHON_PACKAGE "" CACHE PATH "Path to generated inference engine python package.")

function(build_docs)
    find_package(Doxygen REQUIRED dot)
    find_package(PythonInterp 3 REQUIRED)
    find_package(LATEX REQUIRED)

    find_program(DOXYREST_EXECUTABLE NAMES doxyrest)
    if (NOT DOXYREST_EXECUTABLE)
        message(FATAL_ERROR "No doxyrest found. Documentation output is not available")
    endif()

    set(DOCS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(DOCS_SOURCE_DIR "${OpenVINO_MAIN_SOURCE_DIR}/docs")
    set(DOXYGEN_DIR "${DOCS_SOURCE_DIR}/doxygen")

    set(OPENVINO_BIN_DIR "${OpenVINO_MAIN_SOURCE_DIR}/bin/intel64/${CMAKE_BUILD_TYPE}/")
    set(NGRAPH_PYTHON_PACKAGE "${OPENVINO_BIN_DIR}/python_api/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/ngraph")
    set(IE_PYTHON_PACKAGE "${OPENVINO_BIN_DIR}/python_api/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/openvino")

    # API INPUT
    
    set(NGRAPH_DIR "${OpenVINO_MAIN_SOURCE_DIR}/ngraph")

    # markdown docs
    set(MARKDOWN_INPUT "${DOCS_BUILD_DIR}")
    
    # IE C++ API
    set(IE_SOURCE_DIR "${OpenVINO_MAIN_SOURCE_DIR}/inference-engine")
    
    # IE C API
    set(IE_C_API "${IE_SOURCE_DIR}/ie_bridges/c/include")
    
    # IE Python API
    set(IE_PYTHON_API_IN "${IE_SOURCE_DIR}/ie_bridges/python/src/openvino/inference_engine/ie_api.pyx")
    set(IE_PYTHON_API_OUT "${DOCS_BUILD_DIR}/python_api/ie_api.pyx")
    configure_file(${IE_PYTHON_API_IN} ${IE_PYTHON_API_OUT} @ONLY)
    
    # Preprocessing scripts
    set(DOXY_MD_FILTER "${DOXYGEN_DIR}/doxy_md_filter.py")
    set(PYNGRAPH_REF_SCRIPT "${DOXYGEN_DIR}/pyngraph_ref.py")
    set(DOXY_LOG_SCRIPT "${DOXYGEN_DIR}/log.py")
    set(PYX_FILTER "${DOXYGEN_DIR}/pyx_filter.py")
    set(PREPARE_XML_SCRIPT "${DOXYGEN_DIR}/prepare_xml.py")
    set(COPY_IMAGES_SCRIPT "${DOXYGEN_DIR}/copy_images.py")
    set(DOXYGEN_MAPPING_SCRIPT "${DOXYGEN_DIR}/create_mapping.py")
    set(DOXYGEN_MAPPING_FILE "${DOCS_BUILD_DIR}/mapping.json")

    # out dirs
    set(XML_OUTPUT "${DOCS_BUILD_DIR}/xml")
    set(RST_OUTPUT "${DOCS_BUILD_DIR}/rst")
    set(SPHINX_OUTPUT "${DOCS_BUILD_DIR}/_build")

    # Sphinx folders, doxyrest templates and config
    set(SPHINX_CONF_IN "${DOCS_SOURCE_DIR}/conf.py")
    set(SPHINX_CONF_OUT "${RST_OUTPUT}/conf.py")
    set(SPHINX_STATIC_IN "${DOCS_SOURCE_DIR}/_static")
    set(SPHINX_STATIC_OUT "${RST_OUTPUT}/_static")
    set(SPHINX_TEMPLATES_IN "${DOCS_SOURCE_DIR}/_templates")
    set(SPHINX_TEMPLATES_OUT "${RST_OUTPUT}/_templates")
    set(SPHINX_THEMES_IN "${DOCS_SOURCE_DIR}/_themes")
    set(SPHINX_THEMES_OUT "${RST_OUTPUT}/_themes")
    set(SPHINX_INDEX_IN "${DOCS_SOURCE_DIR}/index.rst")
    set(SPHINX_INDEX_OUT "${RST_OUTPUT}/index.rst")
    set(RST_FILES_IN "${DOCS_SOURCE_DIR}/rst_files")
    set(RST_FILES_OUT "${RST_OUTPUT}/api")
    set(DOXYREST_IN "${DOCS_SOURCE_DIR}/doxyrest")
    set(DOXYREST_OUT "${DOCS_BUILD_DIR}/doxyrest")
    set(DOXYREST_SPHINX_IN "${DOCS_SOURCE_DIR}/doxyrest-sphinx")
    set(DOXYREST_SPHINX_OUT "${RST_OUTPUT}/doxyrest-sphinx")
    set(DOXYREST_CONFIG_IN "${DOCS_SOURCE_DIR}/doxyrest-config.lua")
    set(DOXYREST_CONFIG_OUT "${DOCS_BUILD_DIR}/doxyrest-config.lua")
    configure_file(${DOXYREST_CONFIG_IN} ${DOXYREST_CONFIG_OUT} @ONLY)
    configure_file(${SPHINX_CONF_IN} ${SPHINX_CONF_OUT} @ONLY)

    # ie python package
    list(APPEND commands COMMAND sphinx-apidoc "${IE_PYTHON_PACKAGE}" -o "${RST_OUTPUT}/ie_python_api")
    
    # ngraph python package
    list(APPEND commands COMMAND sphinx-apidoc "${NGRAPH_PYTHON_PACKAGE}" -o "${RST_OUTPUT}/ngraph_python_api")

    # Doxygen config
    set(DOXYFILE_SOURCE "${DOXYGEN_DIR}/Doxyfile.config")
    set(DOXYFILE_BUILD "${DOCS_BUILD_DIR}/Doxyfile.config")
    configure_file(${DOXYFILE_SOURCE} ${DOXYFILE_BUILD} @ONLY)

    list(APPEND commands COMMAND ${PYTHON_EXECUTABLE} ${DOXY_MD_FILTER}
        --input_dir=${OpenVINO_MAIN_SOURCE_DIR}
        --output_dir=${DOCS_BUILD_DIR}/openvino
        --exclude_dir=${CMAKE_CURRENT_BINARY_DIR}
        --exclude_dir=${OpenVINO_MAIN_SOURCE_DIR}/out)

    # include additional repositories

    # openvino notebooks dir
    if(EXISTS "${OPENVINO_NOTEBOOKS_DIR}")
        set(OPENVINO_NOTEBOOKS_DIR_OUT "${RST_OUTPUT}/openvino_notebooks")
        list(APPEND commands
            COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENVINO_NOTEBOOKS_DIR_OUT}
        )
        list(APPEND commands
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${OPENVINO_NOTEBOOKS_DIR} ${OPENVINO_NOTEBOOKS_DIR_OUT}
        )
    endif()

    # omz doc files
    if(EXISTS "${OMZ_DOCS_DIR}")
        get_filename_component(OMZ_DOCS_DIR "${OMZ_DOCS_DIR}" ABSOLUTE)
        list(APPEND commands
        COMMAND ${PYTHON_EXECUTABLE} ${OMZ_DOCS_DIR}/ci/prepare-documentation.py ${OMZ_DOCS_DIR}/omz)
        list(APPEND commands COMMAND ${PYTHON_EXECUTABLE} ${DOXY_MD_FILTER}
            --input_dir=${OMZ_DOCS_DIR}/omz
            --output_dir=${DOCS_BUILD_DIR}/omz)
    endif()

    # workbench doc files
    if(EXISTS "${WORKBENCH_DOCS_DIR}")
        get_filename_component(WORKBENCH_DOCS_DIR "${WORKBENCH_DOCS_DIR}" ABSOLUTE)

        list(APPEND commands COMMAND ${PYTHON_EXECUTABLE} ${DOXY_MD_FILTER}
            --input_dir=${WORKBENCH_DOCS_DIR}
            --output_dir=${DOCS_BUILD_DIR}/workbench)
    endif()

    # pot doc files
    if(EXISTS "${POT_DOCS_DIR}")
        get_filename_component(POT_DOCS_DIR "${POT_DOCS_DIR}" ABSOLUTE)

        list(APPEND commands COMMAND ${PYTHON_EXECUTABLE} ${DOXY_MD_FILTER}
        --input_dir=${POT_DOCS_DIR}
        --output_dir=${DOCS_BUILD_DIR}/pot)
    endif()

    # gst doc files
    if(EXISTS "${GST_DOCS_DIR}")
        get_filename_component(GST_DOCS_DIR "${GST_DOCS_DIR}" ABSOLUTE)
        list(APPEND commands COMMAND ${PYTHON_EXECUTABLE} ${DOXY_MD_FILTER}
        --input_dir=${GST_DOCS_DIR}
        --output_dir=${DOCS_BUILD_DIR}/gst)
    endif()

    add_custom_target(doxygen_xml
                      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generate doxygen XML output"
                      VERBATIM)

    # Preprocess docs
    add_custom_command(TARGET doxygen_xml
                       PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E remove_directory ${XML_OUTPUT}
                       ${commands}
                       WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                       COMMENT "Pre-process docs"
                       VERBATIM)

    # Post-process docs
    add_custom_command(TARGET doxygen_xml
                       POST_BUILD
                       COMMAND ${PYTHON_EXECUTABLE} ${PREPARE_XML_SCRIPT} ${XML_OUTPUT}
                       COMMENT "Prepare xml"
                       VERBATIM)

    add_custom_target(sphinx_docs
                      DEPENDS doxygen_xml
                      COMMAND sphinx-build -b html ${RST_OUTPUT} ${SPHINX_OUTPUT}
                      WORKING_DIRECTORY ${RST_OUTPUT}
                      VERBATIM)

    set_target_properties(doxygen_xml sphinx_docs
                          PROPERTIES FOLDER docs)

    add_custom_command(TARGET sphinx_docs
                       PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYREST_IN} ${DOXYREST_OUT}
                       COMMAND ${DOXYREST_EXECUTABLE} -c ${DOXYREST_CONFIG_OUT}
                       COMMAND ${PYTHON_EXECUTABLE} ${COPY_IMAGES_SCRIPT} ${XML_OUTPUT} ${RST_OUTPUT}
                       COMMAND ${PYTHON_EXECUTABLE} ${DOXYGEN_MAPPING_SCRIPT} ${XML_OUTPUT} ${DOCS_BUILD_DIR} ${OpenVINO_MAIN_SOURCE_DIR}/../
                       COMMAND ${CMAKE_COMMAND} -E copy ${SPHINX_INDEX_IN} ${SPHINX_INDEX_OUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${RST_FILES_IN} ${RST_FILES_OUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYREST_IN} ${DOXYREST_OUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYREST_SPHINX_IN} ${DOXYREST_SPHINX_OUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${SPHINX_STATIC_IN} ${SPHINX_STATIC_OUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${SPHINX_TEMPLATES_IN} ${SPHINX_TEMPLATES_OUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${SPHINX_THEMES_IN} ${SPHINX_THEMES_OUT}
                       WORKING_DIRECTORY ${RST_OUTPUT}
                       VERBATIM)

    # add_custom_command(TARGET sphinx_docs
    #                    POST_BUILD
    #                    COMMAND sphinx-build -b linkcheck ${RST_OUTPUT} ${SPHINX_OUTPUT}
    #                    COMMENT "Linkchecker"
    #                    WORKING_DIRECTORY ${RST_OUTPUT}
    #                    VERBATIM)


    find_program(browser NAMES xdg-open)
    if(browser)
        add_custom_target(ie_docs_open
                          COMMAND ${browser} "${SPHINX_OUTPUT}/index.html"
                          DEPENDS sphinx_docs
                          COMMENT "Open OpenVINO documentation"
                          VERBATIM)
        set_target_properties(ie_docs_open PROPERTIES FOLDER docs)
    endif()
endfunction()

if(ENABLE_DOCS)
    build_docs()
endif()
