# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(snippets)

function(build_docs)
    find_package(Doxygen REQUIRED dot)
    find_package(LATEX REQUIRED)

    find_program(DOXYREST_EXECUTABLE NAMES doxyrest)
    if (NOT DOXYREST_EXECUTABLE)
        message(FATAL_ERROR "No doxyrest found. Documentation output is not available")
    endif()

    set(DOCS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(DOCS_SOURCE_DIR "${OpenVINO_SOURCE_DIR}/docs")
    set(SCRIPTS_DIR "${DOCS_SOURCE_DIR}/scripts")

    # markdown docs
    set(MARKDOWN_INPUT "${DOCS_BUILD_DIR}")

    # Preprocessing scripts
    set(DOXY_MD_FILTER "${SCRIPTS_DIR}/doxy_md_filter.py")
    set(PREPARE_XML_SCRIPT "${SCRIPTS_DIR}/prepare_xml.py")
    set(REMOVE_XML_SCRIPT "${SCRIPTS_DIR}/remove_xml.py")
    set(COPY_IMAGES_SCRIPT "${SCRIPTS_DIR}/copy_images.py")
    set(DOC_TEST_DIR "${SCRIPTS_DIR}/tests")
    set(DOXYGEN_MAPPING_SCRIPT "${SCRIPTS_DIR}/create_mapping.py")

    # out dirs
    set(XML_OUTPUT "${DOCS_BUILD_DIR}/xml")
    set(RST_OUTPUT "${DOCS_BUILD_DIR}/rst")
    set(SPHINX_OUTPUT "${DOCS_BUILD_DIR}/_build")

    # Sphinx folders, doxyrest templates and config
    set(DOXYREST_IN "${DOCS_SOURCE_DIR}/doxyrest")
    set(DOXYREST_OUT "${DOCS_BUILD_DIR}/doxyrest")
    set(DOXYREST_CONFIG_IN "${DOCS_SOURCE_DIR}/doxyrest-config.lua")
    set(DOXYREST_CONFIG_OUT "${DOCS_BUILD_DIR}/doxyrest-config.lua")
    configure_file(${DOXYREST_CONFIG_IN} ${DOXYREST_CONFIG_OUT} @ONLY)

    # Doxygen config
    set(DOXYFILE_SOURCE "${DOCS_SOURCE_DIR}/Doxyfile.config")
    set(DOXYFILE_BUILD "${DOCS_BUILD_DIR}/Doxyfile.config")
    configure_file(${DOXYFILE_SOURCE} ${DOXYFILE_BUILD} @ONLY)

    list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${DOXY_MD_FILTER}
        --input_dir=${OpenVINO_SOURCE_DIR}
        --output_dir=${DOCS_BUILD_DIR}/openvino
        --exclude_dir=${DOCS_BUILD_DIR})

    # include additional repositories

    # build with openvino notebooks
    if(ENABLE_OPENVINO_NOTEBOOKS)
        set(NBDOC_SCRIPT "${DOCS_SOURCE_DIR}/nbdoc/nbdoc.py")
        list(PREPEND commands
            COMMAND ${Python3_EXECUTABLE} "${NBDOC_SCRIPT}" "${DOCS_SOURCE_DIR}/notebooks" "${RST_OUTPUT}/notebooks"
        )
    endif()

    if(GRAPH_CSV_DIR)
        set(GRAPH_CSV_DIR_OUT "${RST_OUTPUT}/csv")
        list(APPEND commands
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${GRAPH_CSV_DIR}" "${GRAPH_CSV_DIR_OUT}"
        )
    endif()

    list(APPEND commands
        COMMAND ${CMAKE_COMMAND} -E copy ${API_DOCS_IN}/api_reference.rst ${API_DOCS_OUT}/api_reference.rst
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${API_DOCS_IN}/ie_python_api ${API_DOCS_OUT}/ie_python_api
    )

    # omz doc files
    if(EXISTS "${OMZ_DOCS_DIR}")
        get_filename_component(OMZ_DOCS_DIR "${OMZ_DOCS_DIR}" ABSOLUTE)
        list(APPEND commands
        COMMAND ${Python3_EXECUTABLE} ${OMZ_DOCS_DIR}/ci/prepare-documentation.py ${CMAKE_BINARY_DIR}/open_model_zoo)
        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${DOXY_MD_FILTER}
            --input_dir=${CMAKE_BINARY_DIR}/open_model_zoo
            --output_dir=${DOCS_BUILD_DIR}/open_model_zoo)
    endif()

    # workbench doc files
    if(EXISTS "${WORKBENCH_DOCS_DIR}")
        get_filename_component(WORKBENCH_DOCS_DIR "${WORKBENCH_DOCS_DIR}" ABSOLUTE)

        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${DOXY_MD_FILTER}
            --input_dir=${WORKBENCH_DOCS_DIR}
            --output_dir=${DOCS_BUILD_DIR}/workbench)
    endif()

    # ote doc files
    if(EXISTS "${OTE_DOCS_DIR}")
        get_filename_component(WORKBENCH_DOCS_DIR "${OTE_DOCS_DIR}" ABSOLUTE)

        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${DOXY_MD_FILTER}
            --input_dir=${OTE_DOCS_DIR}
            --output_dir=${DOCS_BUILD_DIR}/ote)
    endif()

    # ovms doc files
    if(EXISTS "${OVMS_DOCS_DIR}")
        get_filename_component(OVMS_DOCS_DIR "${OVMS_DOCS_DIR}" ABSOLUTE)

        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${DOXY_MD_FILTER}
            --input_dir=${OVMS_DOCS_DIR}
            --output_dir=${DOCS_BUILD_DIR}/ovms)
    endif()

    add_custom_target(preprocess_docs
                     COMMENT "Preprocess documentation"
                     VERBATIM)

    # Preprocess docs
    add_custom_command(TARGET preprocess_docs
                       POST_BUILD
                       ${commands}
                       WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                       COMMENT "Preprocess documentation"
                       VERBATIM)

    add_custom_target(doxygen_rst
                      DEPENDS preprocess_docs
                      COMMAND ${Python3_EXECUTABLE} ${REMOVE_XML_SCRIPT} ${XML_OUTPUT}
                      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_BUILD}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      COMMENT "Generate doxygen XML output"
                      VERBATIM)

    # Post-process docs
    add_custom_command(TARGET doxygen_rst
                       POST_BUILD
                       COMMAND ${Python3_EXECUTABLE} ${PREPARE_XML_SCRIPT} ${XML_OUTPUT}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYREST_IN} ${DOXYREST_OUT}
                       COMMAND ${DOXYREST_EXECUTABLE} -c ${DOXYREST_CONFIG_OUT}
                       COMMAND ${Python3_EXECUTABLE} ${COPY_IMAGES_SCRIPT} ${XML_OUTPUT} ${RST_OUTPUT}
                       COMMAND ${Python3_EXECUTABLE} ${DOXYGEN_MAPPING_SCRIPT} ${XML_OUTPUT} ${DOCS_BUILD_DIR} ${OpenVINO_SOURCE_DIR}/../
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${DOXYREST_IN} ${DOXYREST_OUT}
                       COMMENT "Prepare xml"
                       VERBATIM)

    set_target_properties(doxygen_rst
                          PROPERTIES FOLDER docs)

endfunction()

if(ENABLE_DOCS)
    build_docs()
endif()
