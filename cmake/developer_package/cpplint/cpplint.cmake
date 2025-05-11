# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_CPPLINT)
    find_host_package(Python3 QUIET COMPONENTS Interpreter)

    if(NOT Python3_Interpreter_FOUND)
        message(WARNING "Python3 interpreter was not found (required for cpplint check)")
        set(ENABLE_CPPLINT OFF)
    endif()
endif()

if(ENABLE_CPPLINT AND NOT TARGET cpplint_all)
    add_custom_target(cpplint_all ALL)
    set_target_properties(cpplint_all PROPERTIES FOLDER cpplint)
endif()

function(add_cpplint_target TARGET_NAME)
    if(NOT ENABLE_CPPLINT)
        return()
    endif()

    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs FOR_TARGETS FOR_SOURCES EXCLUDE_PATTERNS CUSTOM_FILTERS)
    cmake_parse_arguments(CPPLINT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    foreach(target IN LISTS CPPLINT_FOR_TARGETS)
        get_target_property(target_sources "${target}" SOURCES)
        list(APPEND CPPLINT_FOR_SOURCES ${target_sources})
    endforeach()
    list(REMOVE_DUPLICATES CPPLINT_FOR_SOURCES)

    set(custom_filter "")
    foreach(filter IN LISTS CPPLINT_CUSTOM_FILTERS)
        string(CONCAT custom_filter "${custom_filter}" "," "${filter}")
    endforeach()

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
        file(RELATIVE_PATH source_file_relative_root "${CMAKE_SOURCE_DIR}" "${source_file}")
        set(output_file "${CMAKE_CURRENT_BINARY_DIR}/cpplint/${source_file_relative}.cpplint")
        string(REPLACE ".." "__" output_file "${output_file}")
        get_filename_component(output_dir "${output_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${output_dir}")

        add_custom_command(
            OUTPUT
                "${output_file}"
            COMMAND
                "${CMAKE_COMMAND}"
                -D "Python3_EXECUTABLE=${Python3_EXECUTABLE}"
                -D "CPPLINT_SCRIPT=${OpenVINODeveloperScripts_DIR}/cpplint/cpplint.py"
                -D "INPUT_FILE=${source_file}"
                -D "OUTPUT_FILE=${output_file}"
                -D "WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}"
                -D "SKIP_RETURN_CODE=${ENABLE_CPPLINT_REPORT}"
                -D "CUSTOM_FILTER=${custom_filter}"
                -P "${OpenVINODeveloperScripts_DIR}/cpplint/cpplint_run.cmake"
            DEPENDS
                "${source_file}"
                "${OpenVINODeveloperScripts_DIR}/cpplint/cpplint.py"
                "${OpenVINODeveloperScripts_DIR}/cpplint/cpplint_run.cmake"
            COMMENT
                "[cpplint] ${source_file_relative_root}"
            VERBATIM)

        list(APPEND all_output_files "${output_file}")
    endforeach()

    add_custom_target(${TARGET_NAME} ALL
        DEPENDS ${all_output_files}
        COMMENT "[cpplint] ${TARGET_NAME}")
    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER cpplint)

    if(CPPLINT_FOR_TARGETS)
        foreach(target IN LISTS CPPLINT_FOR_TARGETS)
            add_dependencies(${target} ${TARGET_NAME})
        endforeach()
    endif()

    add_dependencies(cpplint_all ${TARGET_NAME})
endfunction()
