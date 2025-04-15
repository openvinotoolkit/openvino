# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(snippets)

set(ENABLE_CPP_API OFF CACHE BOOL "Build with C/C++ API.")
set(ENABLE_PYTHON_API OFF CACHE BOOL "Build with Python API.")
set(ENABLE_GENAI_API OFF CACHE BOOL "Build with GenAI API.")
set(ENABLE_NOTEBOOKS OFF CACHE BOOL "Build with openvino notebooks.")
set(ENABLE_OVMS OFF CACHE BOOL "Build with ovms.")
set(OVMS_DOCS_DIR "" CACHE PATH "Path to model server documentation dir.")

if(${ENABLE_OVMS} AND (OVMS_DOCS_DIR STREQUAL ""))
    message( FATAL_ERROR "You want to build OVMS, but OVMS_DOCS_DIR variable is empty." )
endif()


function(build_docs)
    find_package(Doxygen REQUIRED dot)
    find_package(LATEX REQUIRED)

    set(DOCS_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(DOCS_SOURCE_DIR "${OpenVINO_SOURCE_DIR}/docs")
    set(ARTICLES_EN_DIR "${OpenVINO_SOURCE_DIR}/docs/articles_en")
    set(SCRIPTS_DIR "${DOCS_SOURCE_DIR}/scripts")

    # Preprocessing scripts
    set(REMOVE_XML_SCRIPT "${SCRIPTS_DIR}/remove_xml.py")
    set(FILE_HELPER_SCRIPT "${SCRIPTS_DIR}/filehelper.py")
    set(ARTICLES_HELPER_SCRIPT "${SCRIPTS_DIR}/articles_helper.py")
    set(COPY_IMAGES_SCRIPT "${SCRIPTS_DIR}/copy_images.py")
    set(DOXYGEN_MAPPING_SCRIPT "${SCRIPTS_DIR}/create_mapping.py")
    set(DOCS_MAPPING_SCRIPT "${SCRIPTS_DIR}/create_doc_mapping.py")
    set(BREATHE_APIDOC_SCRIPT "${SCRIPTS_DIR}/apidoc.py")
    set(OV_INSTALLATION_SCRIPT "${SCRIPTS_DIR}/install_appropriate_openvino_version.py")

    # Doxygen/Sphinx setup
    set(DOXYGEN_XML_OUTPUT "${DOCS_BUILD_DIR}/xml")
    set(SPHINX_SETUP_DIR "${DOCS_SOURCE_DIR}/sphinx_setup")
    set(SPHINX_SOURCE_DIR "${DOCS_BUILD_DIR}/sphinx_source")
    set(SPHINX_OUTPUT "${DOCS_BUILD_DIR}/_build")

    list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "STARTED preprocessing OpenVINO articles")
    list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${ARTICLES_HELPER_SCRIPT}
        --filetype=rst
        --input_dir=${ARTICLES_EN_DIR}
        --output_dir=${SPHINX_SOURCE_DIR}
        --exclude_dir=${SPHINX_SOURCE_DIR})
    list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "FINISHED preprocessing OpenVINO articles")

    if(${ENABLE_CPP_API})
        # Doxygen config
        set(DOXYFILE_SOURCE "${DOCS_SOURCE_DIR}/Doxyfile.config")
        set(DOXYFILE_BUILD "${DOCS_BUILD_DIR}/Doxyfile.config")
        configure_file(${DOXYFILE_SOURCE} ${DOXYFILE_BUILD} @ONLY)
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "STARTED preprocessing OpenVINO C/C++ API reference")
        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${REMOVE_XML_SCRIPT} ${DOXYGEN_XML_OUTPUT})
        list(APPEND commands COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_BUILD})
        list(APPEND post_commands COMMAND ${Python3_EXECUTABLE} ${BREATHE_APIDOC_SCRIPT} ${DOXYGEN_XML_OUTPUT} -o "${SPHINX_SOURCE_DIR}/api/c_cpp_api" -m -T -p openvino -g class,group,struct,union,enum)
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "FINISHED preprocessing OpenVINO C/C++ API reference")
    endif()

    if(${ENABLE_PYTHON_API} OR ${ENABLE_GENAI_API})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "STARTED preprocessing OpenVINO Python API")
        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${OV_INSTALLATION_SCRIPT}
        --ov_dir=${SPHINX_SETUP_DIR}
        --python=${Python3_EXECUTABLE}
        --enable_genai=${ENABLE_GENAI_API})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "FINISHED preprocessing OpenVINO Python API")
    endif()

    if(${ENABLE_OVMS})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "STARTED preprocessing OVMS")
        list(APPEND commands COMMAND ${Python3_EXECUTABLE} ${FILE_HELPER_SCRIPT}
        --filetype=md
        --input_dir=${OVMS_DOCS_DIR}
        --output_dir=${SPHINX_SOURCE_DIR}/model-server
        --exclude_dir=${SPHINX_SOURCE_DIR})
        list(APPEND commands COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "FINISHED preprocessing OVMS")
    endif()


    # Preprocess docs
    add_custom_target(preprocess_docs
                      COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_SOURCE_DIR}
                      ${commands}
                      WORKING_DIRECTORY ${DOCS_BUILD_DIR}
                      VERBATIM)

    add_custom_command(TARGET preprocess_docs
                       POST_BUILD
                       COMMAND ${Python3_EXECUTABLE} ${COPY_IMAGES_SCRIPT} ${DOXYGEN_XML_OUTPUT} ${SPHINX_SOURCE_DIR}
                       COMMAND ${Python3_EXECUTABLE} ${DOXYGEN_MAPPING_SCRIPT} ${DOXYGEN_XML_OUTPUT} ${DOCS_BUILD_DIR} ${OpenVINO_SOURCE_DIR}/../
                       COMMAND ${Python3_EXECUTABLE} ${DOCS_MAPPING_SCRIPT} ${DOCS_BUILD_DIR} ${ARTICLES_EN_DIR}
                       COMMAND ${CMAKE_COMMAND} -E copy_directory ${SPHINX_SETUP_DIR} ${SPHINX_SOURCE_DIR}
                       ${post_commands}
                       VERBATIM)

    # Build docs
    add_custom_target(sphinx_docs
                      DEPENDS preprocess_docs
                      COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "STARTED sphinx documentation build"
                      COMMAND sphinx-build -j auto -w ${DOCS_BUILD_DIR}/sphinx.log -b html ${SPHINX_SOURCE_DIR} ${SPHINX_OUTPUT}
                      COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "FINISHED sphinx documentation build"
                      WORKING_DIRECTORY ${SPHINX_SOURCE_DIR}
                      VERBATIM)

    set_target_properties(preprocess_docs sphinx_docs PROPERTIES FOLDER docs)

endfunction()

if(ENABLE_DOCS)
    build_docs()
endif()
