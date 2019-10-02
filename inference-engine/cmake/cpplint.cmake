# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_CPPLINT)
    find_package(PythonInterp 2.7 EXACT)

    if(NOT PYTHONINTERP_FOUND OR NOT PYTHON_VERSION_MAJOR EQUAL 2)
        message(WARNING "Python 2.7 was not found (required for cpplint check)")
        set(ENABLE_CPPLINT OFF)
    endif()
endif()

if(ENABLE_CPPLINT)
    add_custom_target(cpplint_all ALL)
    set(CPPLINT_ALL_OUTPUT_FILES "" CACHE INTERNAL "All cpplint output files")
endif()

function(add_cpplint_target TARGET_NAME)
    if(NOT ENABLE_CPPLINT)
        return()
    endif()

    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs "FOR_TARGETS" "FOR_SOURCES" "EXCLUDE_PATTERNS")
    cmake_parse_arguments(CPPLINT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(target IN LISTS CPPLINT_FOR_TARGETS)
        get_target_property(target_sources "${target}" SOURCES)
        list(APPEND CPPLINT_FOR_SOURCES ${target_sources})
    endforeach()
    list(REMOVE_DUPLICATES CPPLINT_FOR_SOURCES)

    set(all_output_files "")
    foreach(source_file IN LISTS CPPLINT_FOR_SOURCES)
        set(exclude FALSE)
        foreach(pattern IN LISTS CPPLINT_EXCLUDE_PATTERNS)
            if(source_file MATCHES "${pattern}")
                set(exclude ON)
                break()
            endif()
        endforeach()

        if(exclude)
            continue()
        endif()

        # ignore object libraries
        if(NOT EXISTS "${source_file}")
            continue()
        endif()

        file(RELATIVE_PATH source_file_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${source_file}")
        set(output_file "${CMAKE_CURRENT_BINARY_DIR}/cpplint/${source_file_relative}.cpplint")
        string(REPLACE ".." "__" output_file "${output_file}")
        get_filename_component(output_dir "${output_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${output_dir}")

        add_custom_command(
            OUTPUT
                "${output_file}"
            COMMAND
                "${CMAKE_COMMAND}"
                -D "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
                -D "CPPLINT_SCRIPT=${IE_MAIN_SOURCE_DIR}/scripts/cpplint.py"
                -D "INPUT_FILE=${source_file}"
                -D "OUTPUT_FILE=${output_file}"
                -D "WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}"
                -D "SKIP_RETURN_CODE=${ENABLE_CPPLINT_REPORT}"
                -P "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_run.cmake"
            DEPENDS
                "${source_file}"
                "${IE_MAIN_SOURCE_DIR}/scripts/cpplint.py"
                "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_run.cmake"
            COMMENT
                "[cpplint] ${source_file}"
            VERBATIM)

        list(APPEND all_output_files "${output_file}")
    endforeach()

    set(CPPLINT_ALL_OUTPUT_FILES
        ${CPPLINT_ALL_OUTPUT_FILES} ${all_output_files}
        CACHE INTERNAL
        "All cpplint output files")

    add_custom_target(${TARGET_NAME} ALL
        DEPENDS ${all_output_files}
        COMMENT "[cpplint] ${TARGET_NAME}")

    if(CPPLINT_FOR_TARGETS)
        foreach(target IN LISTS CPPLINT_FOR_TARGETS)
            add_dependencies(${target} ${TARGET_NAME})
        endforeach()
    endif()

    add_dependencies(cpplint_all ${TARGET_NAME})
endfunction()

function(add_cpplint_report_target)
    if(NOT ENABLE_CPPLINT OR NOT ENABLE_CPPLINT_REPORT)
        return()
    endif()

    set(cpplint_output_file "${CMAKE_BINARY_DIR}/cpplint/final_output.cpplint")
    add_custom_command(
        OUTPUT
            "${cpplint_output_file}"
        COMMAND
            "${CMAKE_COMMAND}"
            -D "FINAL_OUTPUT_FILE=${cpplint_output_file}"
            -D "OUTPUT_FILES=${CPPLINT_ALL_OUTPUT_FILES}"
            -P "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_merge.cmake"
        DEPENDS
            ${CPPLINT_ALL_OUTPUT_FILES}
            "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_merge.cmake"
        COMMENT
            "[cpplint] Merge all output files"
        VERBATIM)

    set(cppcheck_output_file "${CMAKE_BINARY_DIR}/cpplint/cpplint-cppcheck-result.xml")
    add_custom_command(
        OUTPUT
            "${cppcheck_output_file}"
        COMMAND
            "${CMAKE_COMMAND}"
            -D "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
            -D "CONVERT_SCRIPT=${IE_MAIN_SOURCE_DIR}/scripts/cpplint_to_cppcheckxml.py"
            -D "INPUT_FILE=${cpplint_output_file}"
            -D "OUTPUT_FILE=${cppcheck_output_file}"
            -P "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_to_cppcheck_xml.cmake"
        DEPENDS
            "${cpplint_output_file}"
            "${IE_MAIN_SOURCE_DIR}/scripts/cpplint_to_cppcheckxml.py"
            "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_to_cppcheck_xml.cmake"
        COMMENT
            "[cpplint] Convert to cppcheck XML format"
        VERBATIM)

    set(report_dir "${IE_MAIN_SOURCE_DIR}/report/cpplint")
    set(html_output_file "${report_dir}/index.html")
    add_custom_command(
        OUTPUT
            "${html_output_file}"
        COMMAND
            "${CMAKE_COMMAND}"
            -D "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
            -D "CONVERT_SCRIPT=${IE_MAIN_SOURCE_DIR}/scripts/cppcheck-htmlreport.py"
            -D "INPUT_FILE=${cppcheck_output_file}"
            -D "REPORT_DIR=${report_dir}"
            -D "SOURCE_DIR=${IE_MAIN_SOURCE_DIR}"
            -D "TITLE=${CMAKE_PROJECT_NAME}"
            -P "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_html.cmake"
        DEPENDS
            "${cppcheck_output_file}"
            "${IE_MAIN_SOURCE_DIR}/scripts/cppcheck-htmlreport.py"
            "${IE_MAIN_SOURCE_DIR}/cmake/cpplint_html.cmake"
        COMMENT
            "[cpplint] Generate HTML report"
        VERBATIM)

    add_custom_target(cpplint_report
        DEPENDS "${html_output_file}"
        COMMENT "[cpplint] Generate report")
endfunction()
