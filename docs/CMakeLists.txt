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

    # Preprocessing scripts
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

    add_custom_target(doxygen_rst
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