# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT COMMAND ie_check_pip_package)
    message(FATAL_ERROR "ncc_naming_style.cmake must be included after ie_check_pip_package")
endif()

# find_host_package(LLVM QUIET)
# if(NOT LLVN_FOUND)
#     message(WARNING "LLVN was not found (required for ncc naming style check)")
#     set(ENABLE_NCC_STYLE OFF)
# endif()

find_package(PythonInterp 3 QUIET)
if(NOT PYTHONINTERP_FOUND)
    message(WARNING "Python3 interpreter was not found (required for ncc naming style check)")
    set(ENABLE_NCC_STYLE OFF)
endif()

if(ENABLE_NCC_STYLE AND NOT TARGET ncc_all)
    add_custom_target(ncc_all ALL)
    set_target_properties(ncc_all PROPERTIES FOLDER ncc_naming_style)
endif()

# check python requirements

set(ncc_style_dir "${IEDevScripts_DIR}/ncc_naming_style")
set(req_file "${ncc_style_dir}/requirements_dev.txt")
file(STRINGS ${req_file} req_lines)

foreach(req IN LISTS req_lines)
    ie_check_pip_package(${req} STATUS)
endforeach()

set(ncc_script_dir "${ncc_style_dir}/ncc/")
set(ncc_script_py "${ncc_style_dir}/ncc/ncc.py")

if(NOT EXISTS ${ncc_script_py})
    message(WARNING "ncc.py is not downloaded via submodule")
    set(ENABLE_NCC_STYLE OFF)
endif()

#
# ov_ncc_naming_style(TARGET_NAME target_name
#                     INCLUDE_DIRECTORY dir
#                     [ADDITIONAL_INCLUDE_DIRECTORIES dir1 dir2 ..])
#
# TARGET_NAME - name of the target
# INCLUDE_DIRECTORY - directory to check headers from
# ADDITIONAL_INCLUDE_DIRECTORIES - additional include directories used in checked headers
#
function(ov_ncc_naming_style)
    if(NOT ENABLE_NCC_STYLE)
        return()
    endif()

    cmake_parse_arguments(NCC_STYLE ""
        "TARGET_NAME;INCLUDE_DIRECTORY" "ADDITIONAL_INCLUDE_DIRECTORIES" ${ARGN})

    file(GLOB_RECURSE headers
         RELATIVE "${NCC_STYLE_INCLUDE_DIRECTORY}"
         "${NCC_STYLE_INCLUDE_DIRECTORY}/*.hpp")

    set(new_pythonpath "${ncc_script_dir}:$ENV{PYTHOPATH}")
    list(APPEND ADDITIONAL_INCLUDE_DIRECTORIES "${NCC_STYLE_INCLUDE_DIRECTORY}")

    foreach(header IN LISTS headers)
        set(output_file "${CMAKE_CURRENT_BINARY_DIR}/ncc_naming_style/${header}.ncc_style")
        set(full_header_path "${NCC_STYLE_INCLUDE_DIRECTORY}/${header}")

        add_custom_command(
            OUTPUT
                ${output_file}
            COMMAND
                "${CMAKE_COMMAND}" -E env PYTHONPATH=${new_pythonpath}
                "${CMAKE_COMMAND}"
                -D "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
                -D "NCC_PY_SCRIPT=${ncc_style_dir}/ncc_wrapper.py"
                -D "INPUT_FILE=${full_header_path}"
                -D "OUTPUT_FILE=${output_file}"
                -D "STYLE_FILE=${ncc_style_dir}/openvino.style"
                -D "ADDITIONAL_INCLUDE_DIRECTORIES=${ADDITIONAL_INCLUDE_DIRECTORIES}"
                -P "${ncc_style_dir}/ncc_run.cmake"
            DEPENDS
                "${full_header_path}"
                "${ncc_script_py}"
                "${ncc_style_dir}/ncc_wrapper.py"
                "${ncc_style_dir}/ncc_run.cmake"
            COMMENT
                "[ncc naming style] ${header}"
            VERBATIM)
        list(APPEND output_files ${output_file})
    endforeach()

    add_custom_target(${NCC_STYLE_TARGET_NAME}_ncc_check
        DEPENDS ${output_files}
        COMMENT "[ncc naming style] ${NCC_STYLE_TARGET_NAME}")

    add_dependencies(ncc_all ${NCC_STYLE_TARGET_NAME}_ncc_check)
endfunction()
