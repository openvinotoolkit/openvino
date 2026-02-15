# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_COVERAGE_GCDA_DATA_DIRECTORY "${CMAKE_BINARY_DIR}")

if (NOT TARGET ov_coverage)
    add_custom_target(ov_coverage)
    set_target_properties(ov_coverage PROPERTIES FOLDER coverage)
endif()

if(NOT TARGET ov_coverage_clean)
    add_custom_target(ov_coverage_clean)
    set_target_properties(ov_coverage_clean PROPERTIES FOLDER coverage)
endif()


set(OV_COVERAGE_REPORTS "${CMAKE_BINARY_DIR}/coverage")
set(OV_COVERAGE_SCRIPT_DIR "${OpenVINODeveloperScripts_DIR}/coverage")

include(CMakeParseArguments)

#
# ov_coverage_clean(REPOSITORY <repo> DIRECTORY <dir>)
#
function(ov_coverage_clean)
    cmake_parse_arguments(OV_COVERAGE "" "REPOSITORY;DIRECTORY" "" ${ARGN})

    add_custom_target(ov_coverage_zerocounters_${OV_COVERAGE_REPOSITORY}
                      COMMAND lcov --zerocounters --quiet
                                   --directory "${OV_COVERAGE_DIRECTORY}"
                      COMMENT "Add zero counters for coverage for ${OV_COVERAGE_REPOSITORY}"
                      VERBATIM)

    add_custom_target(ov_coverage_clean_${OV_COVERAGE_REPOSITORY}
                      COMMAND ${CMAKE_COMMAND}
                        -D "OV_COVERAGE_REPORTS=${OV_COVERAGE_REPORTS}"
                        -D "OV_COVERAGE_DIRECTORY=${OV_COVERAGE_DIRECTORY}"
                        -D "CMAKE_BINARY_DIRECTORY=${CMAKE_BINARY_DIR}"
                        -D "CMAKE_SOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
                        -P "${OV_COVERAGE_SCRIPT_DIR}/coverage_clean.cmake"
                      COMMENT "Clean previously created HTML report files for ${OV_COVERAGE_REPOSITORY}"
                      DEPENDS "${OV_COVERAGE_SCRIPT_DIR}/coverage_clean.cmake"
                      VERBATIM)

    set_target_properties(ov_coverage_zerocounters_${OV_COVERAGE_REPOSITORY}
                          ov_coverage_clean_${OV_COVERAGE_REPOSITORY}
                          PROPERTIES FOLDER coverage)

    add_dependencies(ov_coverage_clean ov_coverage_zerocounters_${OV_COVERAGE_REPOSITORY}
                                       ov_coverage_clean_${OV_COVERAGE_REPOSITORY})
endfunction()

#
# ov_coverage_capture(INFO_FILE <info_file>
#                     BASE_DIRECTORY <base dir>
#                     DIRECTORY <gcda dir>
#                     EXCLUDE_PATTERNS exclude_patterns,...)
#
function(ov_coverage_capture)
    cmake_parse_arguments(OV_COVERAGE "" "INFO_FILE;BASE_DIRECTORY;DIRECTORY" "EXCLUDE_PATTERNS" ${ARGN})

    set(output_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INFO_FILE}.info")
    set(output_base_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INFO_FILE}_base.info")
    set(output_tests_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INFO_FILE}_tests.info")

    add_custom_command(OUTPUT ${output_base_file}
                       COMMAND ${CMAKE_COMMAND} -E make_directory "${OV_COVERAGE_REPORTS}"
                       COMMAND lcov --no-external --capture --initial --quiet
                                    --directory "${OV_COVERAGE_DIRECTORY}"
                                    --base-directory "${OV_COVERAGE_BASE_DIRECTORY}"
                                    --output-file ${output_base_file}
                       COMMENT "Capture initial coverage data ${OV_COVERAGE_INFO_FILE}"
                       VERBATIM)

    add_custom_command(OUTPUT ${output_tests_file}
                       COMMAND ${CMAKE_COMMAND} -E make_directory "${OV_COVERAGE_REPORTS}"
                       COMMAND lcov --no-external --capture --quiet
                                    --directory "${OV_COVERAGE_DIRECTORY}"
                                    --base-directory "${OV_COVERAGE_BASE_DIRECTORY}"
                                    --output-file ${output_tests_file}
                       COMMENT "Capture test coverage data ${OV_COVERAGE_INFO_FILE}"
                       VERBATIM)

    if (OV_COVERAGE_EXCLUDE_PATTERNS)
        set(out_suf ".tmp")
    endif()

    add_custom_command(OUTPUT ${output_file}${out_suf}
                       COMMAND ${CMAKE_COMMAND}
                               -D "OV_COVERAGE_OUTPUT_FILE=${output_file}${out_suf}"
                               -D "OV_COVERAGE_INPUT_FILES=${output_base_file};${output_tests_file}"
                               -P "${OV_COVERAGE_SCRIPT_DIR}/coverage_merge.cmake"
                       COMMENT "Generate total coverage data ${OV_COVERAGE_INFO_FILE}"
                       DEPENDS ${output_base_file} ${output_tests_file}
                       VERBATIM)

    if (OV_COVERAGE_EXCLUDE_PATTERNS)
        set(commands lcov --quiet)
        foreach(pattern IN LISTS OV_COVERAGE_EXCLUDE_PATTERNS)
            list(APPEND commands --remove ${output_file}${out_suf} ${pattern})
        endforeach()
        list(APPEND commands --output-file ${output_file})

        add_custom_command(OUTPUT ${output_file}
            COMMAND ${commands}
            COMMENT "Exclude patterns from report ${OV_COVERAGE_OUTPUT}"
            DEPENDS ${output_file}${out_suf}
            VERBATIM)
    endif()

    add_custom_target(ov_coverage_${OV_COVERAGE_INFO_FILE}_info
                      DEPENDS ${output_file})
    set_target_properties(ov_coverage_${OV_COVERAGE_INFO_FILE}_info
                          PROPERTIES FOLDER coverage)
endfunction()

#
# ov_coverage_extract(INPUT <info_file> OUTPUT <output_file> PATTERNS <patterns ...>)
#
function(ov_coverage_extract)
    cmake_parse_arguments(OV_COVERAGE "" "INPUT;OUTPUT" "PATTERNS" ${ARGN})

    set(input_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INPUT}.info")
    set(output_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_OUTPUT}.info")

    set(commands lcov --quiet)
    foreach(pattern IN LISTS OV_COVERAGE_PATTERNS)
        list(APPEND commands --extract ${input_file} ${pattern})
    endforeach()
    list(APPEND commands --output-file ${output_file})

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${commands}
                       COMMENT "Generate coverage data ${OV_COVERAGE_OUTPUT}"
                       DEPENDS ${input_file}
                       VERBATIM)
    add_custom_target(ov_coverage_${OV_COVERAGE_OUTPUT}_info
                      DEPENDS ${output_file})
    set_target_properties(ov_coverage_${OV_COVERAGE_OUTPUT}_info
                          PROPERTIES FOLDER coverage)

    add_dependencies(ov_coverage_${OV_COVERAGE_OUTPUT}_info ov_coverage_${OV_COVERAGE_INPUT}_info)
endfunction()

#
# ov_coverage_genhtml(INFO_FILE <info_file> PREFIX <prefix>)
#
function(ov_coverage_genhtml)
    cmake_parse_arguments(OV_COVERAGE "" "INFO_FILE;PREFIX" "" ${ARGN})

    set(input_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INFO_FILE}.info")
    set(output_directory "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INFO_FILE}")

    add_custom_command(OUTPUT "${output_directory}/index.html"
                       COMMAND [ -s ${input_file}  ] && genhtml ${input_file} --title "${OV_COVERAGE_INFO_FILE}" --legend
                                       --no-branch-coverage --demangle-cpp
                                       --output-directory "${output_directory}"
                                       --num-spaces 4 --quiet
                                       --prefix "${OV_COVERAGE_PREFIX}" || echo "Skip ${input_file}"
                       DEPENDS ${input_file}
                       COMMENT "Generate HTML report for ${OV_COVERAGE_INFO_FILE}"
                       VERBATIM)
    add_custom_target(ov_coverage_${OV_COVERAGE_INFO_FILE}_genhtml
                      DEPENDS "${output_directory}/index.html")
    set_target_properties(ov_coverage_${OV_COVERAGE_INFO_FILE}_genhtml
                          PROPERTIES FOLDER coverage)

    add_dependencies(ov_coverage_${OV_COVERAGE_INFO_FILE}_genhtml ov_coverage_${OV_COVERAGE_INFO_FILE}_info)
    add_dependencies(ov_coverage ov_coverage_${OV_COVERAGE_INFO_FILE}_genhtml)
endfunction()

#
# ov_coverage_remove(INPUT <info_file> OUTPUT <output_file> PATTERNS <patterns ...>)
#
function(ov_coverage_remove)
    cmake_parse_arguments(OV_COVERAGE "" "INPUT;OUTPUT" "PATTERNS" ${ARGN})

    set(input_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_INPUT}.info")
    set(output_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_OUTPUT}.info")

    set(commands lcov --quiet)
    foreach(pattern IN LISTS OV_COVERAGE_PATTERNS)
        list(APPEND commands --remove ${input_file} ${pattern})
    endforeach()
    list(APPEND commands --output-file ${output_file})

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${commands}
                       COMMENT "Generate coverage data ${OV_COVERAGE_OUTPUT}"
                       DEPENDS ${input_file}
                       VERBATIM)
    add_custom_target(ov_coverage_${OV_COVERAGE_OUTPUT}_info
                      DEPENDS ${output_file})
    set_target_properties(ov_coverage_${OV_COVERAGE_OUTPUT}_info
                          PROPERTIES FOLDER coverage)

    add_dependencies(ov_coverage_${OV_COVERAGE_OUTPUT}_info ov_coverage_${OV_COVERAGE_INPUT}_info)
endfunction()

#
# ov_coverage_merge(OUTPUT <output file> INPUTS <input files ...>)
#
function(ov_coverage_merge)
    cmake_parse_arguments(OV_COVERAGE "" "OUTPUT" "INPUTS" ${ARGN})

    set(output_file "${OV_COVERAGE_REPORTS}/${OV_COVERAGE_OUTPUT}.info")
    foreach(input_info_file IN LISTS OV_COVERAGE_INPUTS)
        set(input_file ${OV_COVERAGE_REPORTS}/${input_info_file}.info)
        list(APPEND dependencies ov_coverage_${input_info_file}_info)
        list(APPEND input_files ${input_file})
    endforeach()

    add_custom_command(OUTPUT ${output_file}
                       COMMAND ${CMAKE_COMMAND}
                               -D "OV_COVERAGE_OUTPUT_FILE=${output_file}"
                               -D "OV_COVERAGE_INPUT_FILES=${input_files}"
                               -P "${OV_COVERAGE_SCRIPT_DIR}/coverage_merge.cmake"
                       COMMENT "Generate coverage data ${OV_COVERAGE_OUTPUT}"
                       DEPENDS ${input_files}
                       VERBATIM)
    add_custom_target(ov_coverage_${OV_COVERAGE_OUTPUT}_info
                      DEPENDS ${output_file})
    set_target_properties(ov_coverage_${OV_COVERAGE_OUTPUT}_info
                          PROPERTIES FOLDER coverage)

    add_dependencies(ov_coverage_${OV_COVERAGE_OUTPUT}_info ${dependencies})
endfunction()
