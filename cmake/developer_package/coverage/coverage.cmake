# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_COVERAGE_GCDA_DATA_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT TARGET ie_coverage_clean)
    add_custom_target(ie_coverage_clean)
    set_target_properties(ie_coverage_clean PROPERTIES FOLDER coverage)
endif()

if(NOT TARGET ie_coverage_init)
    add_custom_target(ie_coverage_init)
    set_target_properties(ie_coverage_init PROPERTIES FOLDER coverage)
endif()

if(NOT TARGET ie_coverage)
    add_custom_target(ie_coverage)
    set_target_properties(ie_coverage PROPERTIES FOLDER coverage)
endif()

set(IE_COVERAGE_REPORTS "${CMAKE_BINARY_DIR}/coverage")
set(IE_COVERAGE_SCRIPT_DIR "${IEDevScripts_DIR}/coverage")

include(CMakeParseArguments)

#
# ie_coverage_clean(REPOSITORY <repo> DIRECTORY <dir>)
#
function(ie_coverage_clean)
    cmake_parse_arguments(IE_COVERAGE "" "REPOSITORY;DIRECTORY" "" ${ARGN})

    add_custom_target(ie_coverage_zerocounters_${IE_COVERAGE_REPOSITORY}
                      COMMAND lcov --zerocounters --quiet
                                   --directory "${IE_COVERAGE_DIRECTORY}"
                      COMMENT "Add zero counters for coverage for ${IE_COVERAGE_REPOSITORY}"
                      VERBATIM)

    add_custom_target(ie_coverage_clean_${IE_COVERAGE_REPOSITORY}
                      COMMAND ${CMAKE_COMMAND}
                        -D "IE_COVERAGE_REPORTS=${IE_COVERAGE_REPORTS}"
                        -D "IE_COVERAGE_DIRECTORY=${IE_COVERAGE_DIRECTORY}"
                        -D "CMAKE_BINARY_DIRECTORY=${CMAKE_BINARY_DIR}"
                        -D "CMAKE_SOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
                        -P "${IE_COVERAGE_SCRIPT_DIR}/coverage_clean.cmake"
                      COMMENT "Clean previously created HTML report files for ${IE_COVERAGE_REPOSITORY}"
                      DEPENDS "${IE_COVERAGE_SCRIPT_DIR}/coverage_clean.cmake"
                      VERBATIM)

    set_target_properties(ie_coverage_zerocounters_${IE_COVERAGE_REPOSITORY}
                          ie_coverage_clean_${IE_COVERAGE_REPOSITORY}
                          PROPERTIES FOLDER coverage)

    add_dependencies(ie_coverage_clean ie_coverage_zerocounters_${IE_COVERAGE_REPOSITORY}
                                       ie_coverage_clean_${IE_COVERAGE_REPOSITORY})
endfunction()

#
# ie_coverage_capture(INFO_FILE <info_file>
#                     BASE_DIRECTORY <base dir>
#                     DIRECTORY <gcda dir>)
#
function(ie_coverage_capture)
    cmake_parse_arguments(IE_COVERAGE "" "INFO_FILE;BASE_DIRECTORY;DIRECTORY" "" ${ARGN})

    set(output_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INFO_FILE}.info")
    set(output_base_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INFO_FILE}_base.info")
    set(output_tests_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INFO_FILE}_tests.info")

    add_custom_command(OUTPUT ${output_base_file}
                       COMMAND ${CMAKE_COMMAND} -E make_directory "${IE_COVERAGE_REPORTS}"
                       COMMAND lcov --no-external --capture --initial --quiet
                                    --directory "${IE_COVERAGE_DIRECTORY}"
                                    --base-directory "${IE_COVERAGE_BASE_DIRECTORY}"
                                    --output-file ${output_base_file}
                       COMMENT "Capture initial coverage data ${IE_COVERAGE_INFO_FILE}"
                       VERBATIM)

    add_custom_command(OUTPUT ${output_tests_file}
                       COMMAND ${CMAKE_COMMAND} -E make_directory "${IE_COVERAGE_REPORTS}"
                       COMMAND lcov --no-external --capture --quiet
                                    --directory "${IE_COVERAGE_DIRECTORY}"
                                    --base-directory "${IE_COVERAGE_BASE_DIRECTORY}"
                                    --output-file ${output_tests_file}
                       COMMENT "Capture test coverage data ${IE_COVERAGE_INFO_FILE}"
                       VERBATIM)

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${CMAKE_COMMAND}
                               -D "IE_COVERAGE_OUTPUT_FILE=${output_file}"
                               -D "IE_COVERAGE_INPUT_FILES=${output_base_file};${output_tests_file}"
                               -P "${IE_COVERAGE_SCRIPT_DIR}/coverage_merge.cmake"
                       COMMENT "Generate total coverage data ${IE_COVERAGE_INFO_FILE}"
                       DEPENDS ${output_base_file} ${output_tests_file}
                       VERBATIM)

    add_custom_target(ie_coverage_${IE_COVERAGE_INFO_FILE}_info
                      DEPENDS ${output_file})
    set_target_properties(ie_coverage_${IE_COVERAGE_INFO_FILE}_info
                          PROPERTIES FOLDER coverage)
endfunction()

#
# ie_coverage_extract(INPUT <info_file> OUTPUT <output_file> PATTERNS <patterns ...>)
#
function(ie_coverage_extract)
    cmake_parse_arguments(IE_COVERAGE "" "INPUT;OUTPUT" "PATTERNS" ${ARGN})

    set(input_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INPUT}.info")
    set(output_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_OUTPUT}.info")

    set(commands lcov --quiet)
    foreach(pattern IN LISTS IE_COVERAGE_PATTERNS)
        list(APPEND commands --extract ${input_file} ${pattern})
    endforeach()
    list(APPEND commands --output-file ${output_file})

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${commands}
                       COMMENT "Generate coverage data ${IE_COVERAGE_OUTPUT}"
                       DEPENDS ${input_file}
                       VERBATIM)
    add_custom_target(ie_coverage_${IE_COVERAGE_OUTPUT}_info
                      DEPENDS ${output_file})
    set_target_properties(ie_coverage_${IE_COVERAGE_OUTPUT}_info
                          PROPERTIES FOLDER coverage)

    add_dependencies(ie_coverage_${IE_COVERAGE_OUTPUT}_info ie_coverage_${IE_COVERAGE_INPUT}_info)
endfunction()

#
# ie_coverage_remove(INPUT <info_file> OUTPUT <output_file> PATTERNS <patterns ...>)
#
function(ie_coverage_remove)
    cmake_parse_arguments(IE_COVERAGE "" "INPUT;OUTPUT" "PATTERNS" ${ARGN})

    set(input_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INPUT}.info")
    set(output_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_OUTPUT}.info")

    set(commands lcov --quiet)
    foreach(pattern IN LISTS IE_COVERAGE_PATTERNS)
        list(APPEND commands --remove ${input_file} ${pattern})
    endforeach()
    list(APPEND commands --output-file ${output_file})

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${commands}
                       COMMENT "Generate coverage data ${IE_COVERAGE_OUTPUT}"
                       DEPENDS ${input_file}
                       VERBATIM)
    add_custom_target(ie_coverage_${IE_COVERAGE_OUTPUT}_info
                      DEPENDS ${output_file})
    set_target_properties(ie_coverage_${IE_COVERAGE_OUTPUT}_info
                          PROPERTIES FOLDER coverage)

    add_dependencies(ie_coverage_${IE_COVERAGE_OUTPUT}_info ie_coverage_${IE_COVERAGE_INPUT}_info)
endfunction()

#
# ie_coverage_merge(OUTPUT <output file> INPUTS <input files ...>)
#
function(ie_coverage_merge)
    cmake_parse_arguments(IE_COVERAGE "" "OUTPUT" "INPUTS" ${ARGN})

    set(output_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_OUTPUT}.info")
    foreach(input_info_file IN LISTS IE_COVERAGE_INPUTS)
        set(input_file ${IE_COVERAGE_REPORTS}/${input_info_file}.info)
        list(APPEND dependencies ie_coverage_${input_info_file}_info)
        list(APPEND input_files ${input_file})
    endforeach()

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${CMAKE_COMMAND}
                               -D "IE_COVERAGE_OUTPUT_FILE=${output_file}"
                               -D "IE_COVERAGE_INPUT_FILES=${input_files}"
                               -P "${IE_COVERAGE_SCRIPT_DIR}/coverage_merge.cmake"
                       COMMENT "Generate coverage data ${IE_COVERAGE_OUTPUT}"
                       DEPENDS ${input_files}
                       VERBATIM)
    add_custom_target(ie_coverage_${IE_COVERAGE_OUTPUT}_info
                      DEPENDS ${output_file})
    set_target_properties(ie_coverage_${IE_COVERAGE_OUTPUT}_info
                          PROPERTIES FOLDER coverage)

    add_dependencies(ie_coverage_${IE_COVERAGE_OUTPUT}_info ${dependencies})
endfunction()

#
# ie_coverage_genhtml(INFO_FILE <info_file> PREFIX <prefix>)
#
function(ie_coverage_genhtml)
    cmake_parse_arguments(IE_COVERAGE "" "INFO_FILE;PREFIX" "" ${ARGN})

    set(input_file "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INFO_FILE}.info")
    set(output_directory "${IE_COVERAGE_REPORTS}/${IE_COVERAGE_INFO_FILE}")

    add_custom_command(OUTPUT "${output_directory}/index.html"
                       COMMAND genhtml ${input_file} --title "${IE_COVERAGE_INFO_FILE}" --legend
                                       --no-branch-coverage --demangle-cpp
                                       --output-directory "${output_directory}"
                                       --num-spaces 4 --quiet
                                       --prefix "${IE_COVERAGE_PREFIX}"
                       DEPENDS ${input_file}
                       COMMENT "Generate HTML report for ${IE_COVERAGE_INFO_FILE}"
                       VERBATIM)
    add_custom_target(ie_coverage_${IE_COVERAGE_INFO_FILE}_genhtml
                      DEPENDS "${output_directory}/index.html")
    set_target_properties(ie_coverage_${IE_COVERAGE_INFO_FILE}_genhtml
                          PROPERTIES FOLDER coverage)

    add_dependencies(ie_coverage_${IE_COVERAGE_INFO_FILE}_genhtml ie_coverage_${IE_COVERAGE_INFO_FILE}_info)
    add_dependencies(ie_coverage ie_coverage_${IE_COVERAGE_INFO_FILE}_genhtml)
endfunction()
