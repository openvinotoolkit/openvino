# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT COMMAND ov_check_pip_packages)
    message(FATAL_ERROR "Internal error: ncc_naming_style.cmake must be included after ov_check_pip_packages")
endif()

set(ncc_style_dir "${IEDevScripts_DIR}/ncc_naming_style")
set(ncc_style_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/ncc_naming_style")

# find python3

find_host_package(PythonInterp 3 QUIET)
if(NOT PYTHONINTERP_FOUND)
    message(WARNING "Python3 interpreter was not found (required for ncc naming style check)")
    set(ENABLE_NCC_STYLE OFF)
endif()

if(PYTHON_VERSION_MINOR EQUAL 6)
    set(clang_version 10)
elseif(PYTHON_VERSION_MINOR EQUAL 7)
    set(clang_version 11)
elseif(PYTHON_VERSION_MINOR EQUAL 8)
    set(clang_version 12)
elseif(PYTHON_VERSION_MINOR EQUAL 9)
    set(clang_version 12)
elseif(PYTHON_VERSION_MINOR EQUAL 10)
    set(clang_version 14)
else()
    message(WARNING "Cannot suggest clang package for python ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()


if(ENABLE_NCC_STYLE)
    # try to find_package(Clang QUIET)
    # ClangConfig.cmake contains bug that if libclang-XX-dev is not
    # installed, then find_package fails with errors even in QUIET mode
    configure_file("${ncc_style_dir}/try_find_clang.cmake"
                   "${ncc_style_bin_dir}/source/CMakeLists.txt" COPYONLY)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -S "${ncc_style_bin_dir}/source"
                                   -B "${ncc_style_bin_dir}/build"
        RESULT_VARIABLE clang_find_result
        OUTPUT_VARIABLE output_var
        ERROR_VARIABLE error_var)

    if(NOT clang_find_result EQUAL "0")
        message(WARNING "Please, install `apt-get install clang-${clang_version} libclang-${clang_version}-dev` package (required for ncc naming style check)")
        message(TRACE "find_package(Clang) output: ${output_var}")
        message(TRACE "find_package(Clang) error: ${error_var}")
        set(ENABLE_NCC_STYLE OFF)
    endif()
endif()

# Since we were able to find_package(Clang) in a separate process
# let's try to find in current process
if(ENABLE_NCC_STYLE)
    if(APPLE)
        find_host_library(libclang_location NAMES clang
                          PATHS /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib
                          DOC "Path to clang library")
    else()
        find_host_package(Clang QUIET)
    endif()

    if(Clang_FOUND AND TARGET libclang)
        get_target_property(libclang_location libclang LOCATION)
    endif()

    if(NOT libclang_location)
        message(WARNING "clang-${clang_version} libclang-${clang_version}-dev are not found (required for ncc naming style check)")
        set(ENABLE_NCC_STYLE OFF)
    else()
        message(STATUS "Found libclang: ${libclang_location}")
    endif()
endif()

# check python requirements_dev.txt

set(ncc_script_py "${ncc_style_dir}/ncc/ncc.py")

if(NOT EXISTS ${ncc_script_py})
    message(WARNING "ncc.py is not downloaded via submodule")
    set(ENABLE_NCC_STYLE OFF)
endif()

if(ENABLE_NCC_STYLE)
    ov_check_pip_packages(REQUIREMENTS_FILE "${ncc_style_dir}/requirements_dev.txt"
                          RESULT_VAR python_clang_FOUND
                          WARNING_MESSAGE "NCC style check will be unavailable"
                          MESSAGE_MODE WARNING)
    if(NOT python_clang_FOUND)
        # Note: warnings is already thrown by `ov_check_pip_packages`
        set(ENABLE_NCC_STYLE OFF)
    endif()
endif()

# create high-level target

if(ENABLE_NCC_STYLE AND NOT TARGET ncc_all)
    add_custom_target(ncc_all ALL)
    set_target_properties(ncc_all PROPERTIES FOLDER ncc_naming_style)
endif()

#
# ov_ncc_naming_style(FOR_TARGET target_name
#                     SOURCE_DIRECTORY dir
#                     [ADDITIONAL_INCLUDE_DIRECTORIES dir1 dir2 ..]
#                     [DEFINITIONS def1 def2 ..])
#
# FOR_TARGET - name of the target
# SOURCE_DIRECTORY - directory to check sources from
# ADDITIONAL_INCLUDE_DIRECTORIES - additional include directories used in checked headers
# DEFINITIONS - additional definitions passed to preprocessor stage
#
function(ov_ncc_naming_style)
    if(NOT ENABLE_NCC_STYLE)
        return()
    endif()

    cmake_parse_arguments(NCC_STYLE "FAIL"
        "FOR_TARGET;SOURCE_DIRECTORY" "ADDITIONAL_INCLUDE_DIRECTORIES;DEFINITIONS" ${ARGN})

    foreach(var FOR_TARGET SOURCE_DIRECTORY)
        if(NOT DEFINED NCC_STYLE_${var})
            message(FATAL_ERROR "${var} is not defined in ov_ncc_naming_style function")
        endif()
    endforeach()

    file(GLOB_RECURSE sources
         RELATIVE "${NCC_STYLE_SOURCE_DIRECTORY}"
         "${NCC_STYLE_SOURCE_DIRECTORY}/*.hpp"
         "${NCC_STYLE_SOURCE_DIRECTORY}/*.cpp")

    list(APPEND NCC_STYLE_ADDITIONAL_INCLUDE_DIRECTORIES "${NCC_STYLE_SOURCE_DIRECTORY}")
    # without it sources with same name from different directories will map to same .ncc_style target
    file(RELATIVE_PATH source_dir_rel ${CMAKE_SOURCE_DIR} ${NCC_STYLE_SOURCE_DIRECTORY})

    foreach(source IN LISTS sources)
        set(output_file "${ncc_style_bin_dir}/${source_dir_rel}/${source}.ncc_style")
        set(full_source_path "${NCC_STYLE_SOURCE_DIRECTORY}/${source}")

        add_custom_command(
            OUTPUT
                ${output_file}
            COMMAND
                "${CMAKE_COMMAND}"
                -D "PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
                -D "NCC_PY_SCRIPT=${ncc_script_py}"
                -D "INPUT_FILE=${full_source_path}"
                -D "OUTPUT_FILE=${output_file}"
                -D "DEFINITIONS=${NCC_STYLE_DEFINITIONS}"
                -D "CLANG_LIB_PATH=${libclang_location}"
                -D "STYLE_FILE=${ncc_style_dir}/openvino.style"
                -D "ADDITIONAL_INCLUDE_DIRECTORIES=${NCC_STYLE_ADDITIONAL_INCLUDE_DIRECTORIES}"
                -D "EXPECTED_FAIL=${NCC_STYLE_FAIL}"
                -P "${ncc_style_dir}/ncc_run.cmake"
            DEPENDS
                "${full_source_path}"
                "${ncc_style_dir}/openvino.style"
                "${ncc_script_py}"
                "${ncc_style_dir}/ncc_run.cmake"
            COMMENT
                "[ncc naming style] ${source}"
            VERBATIM)
        list(APPEND output_files ${output_file})
    endforeach()

    set(ncc_target ${NCC_STYLE_FOR_TARGET}_ncc_check)
    add_custom_target(${ncc_target}
        DEPENDS ${output_files}
        COMMENT "[ncc naming style] ${NCC_STYLE_FOR_TARGET}")

    add_dependencies(${NCC_STYLE_FOR_TARGET} ${ncc_target})
    add_dependencies(ncc_all ${ncc_target})
endfunction()

if(TARGET ncc_all)
    ov_ncc_naming_style(FOR_TARGET ncc_all
                        SOURCE_DIRECTORY "${ncc_style_dir}/self_check"
                        FAIL)
endif()
