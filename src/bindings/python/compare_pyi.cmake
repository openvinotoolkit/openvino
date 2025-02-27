# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var IN ITEMS generated_pyi_files_location source_pyi_files_location)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

# Find all generated and committed .pyi files
file(GLOB_RECURSE generated_pyi_files ${generated_pyi_files_location}/*.pyi)
file(GLOB_RECURSE committed_pyi_files ${source_pyi_files_location}/*.pyi)

# Check the files count
list(LENGTH generated_pyi_files generated_pyi_files_count)
list(LENGTH committed_pyi_files committed_pyi_files_count)
if(NOT generated_pyi_files_count EQUAL committed_pyi_files_count OR generated_pyi_files_count EQUAL 0)
    message(FATAL_ERROR "The numbers of generated .pyi files (${generated_pyi_files_count}) and committed .pyi files (${committed_pyi_files_count}) are incorrect.")
endif()

foreach(generated_file IN LISTS generated_pyi_files)
    file(RELATIVE_PATH relative_path ${CMAKE_CURRENT_BINARY_DIR}/pyapi_stubs_generated/openvino ${generated_file})
    set(committed_file "${source_pyi_files_location}/${relative_path}")

    # Every file has to have a pair
    if(NOT EXISTS ${committed_file})
        message(FATAL_ERROR "Committed .pyi file not found for generated file: ${generated_file}")
    endif()

    # Use a custom script instead of compare_files (ticket: 163225)
    message(STATUS "python exec: ${python_exec}")
    execute_process(COMMAND ${python_exec} ${source_pyi_files_location}/../../scripts/compare_pyi_files.py ${generated_file} ${committed_file}
        RESULT_VARIABLE compare_result
    )
    if(NOT compare_result EQUAL 0)
        message(FATAL_ERROR "The reference .pyi file ${generated_file} does not match the committed .pyi file ${committed_file}. Please refer to documentation to generate up-to-date .pyi files.")
    endif()
endforeach()
message(STATUS "Python stub files validation successful.")
