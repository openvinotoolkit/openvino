# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT COMMAND ov_check_pip_packages)
    message(FATAL_ERROR "Internal error: ncc_naming_style.cmake must be included after ov_check_pip_packages")
endif()

set(ncc_style_dir "${OpenVINODeveloperScripts_DIR}/ncc_naming_style")
set(ncc_style_bin_dir "${CMAKE_CURRENT_BINARY_DIR}/ncc_naming_style")

# find python3

if(ENABLE_NCC_STYLE)
    find_host_package(Python3 QUIET COMPONENTS Interpreter)
    if(NOT Python3_Interpreter_FOUND)
        message(WARNING "Python3 interpreter was not found (required for ncc naming style check)")
        set(ENABLE_NCC_STYLE OFF)
    endif()
endif()

if(ENABLE_NCC_STYLE)
    if(Python3_VERSION_MINOR EQUAL 6)
        set(clang_version 10)
    elseif(Python3_VERSION_MINOR EQUAL 7)
        set(clang_version 11)
    elseif(Python3_VERSION_MINOR EQUAL 8)
        set(clang_version 12)
    elseif(Python3_VERSION_MINOR EQUAL 9)
        set(clang_version 12)
    elseif(Python3_VERSION_MINOR EQUAL 10)
        set(clang_version 14)
    elseif(Python3_VERSION_MINOR EQUAL 11)
        set(clang_version 14)
    elseif(Python3_VERSION_MINOR EQUAL 12)
        set(clang_version 15)
    else()
        message(WARNING "Cannot suggest clang package for python ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
    endif()
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
    if(CMAKE_HOST_WIN32)
        find_host_program(libclang_location NAMES libclang.dll
                          PATHS $ENV{PATH}
                          NO_CMAKE_FIND_ROOT_PATH)
    elseif(CMAKE_HOST_APPLE)
        set(_old_CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES})
        set(_old_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
        set(CMAKE_FIND_LIBRARY_PREFIXES "lib")
        set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib")
        find_host_library(libclang_location NAMES clang
                          PATHS /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib
                          DOC "Path to clang library"
                          NO_DEFAULT_PATH
                          NO_CMAKE_FIND_ROOT_PATH)
        set(CMAKE_FIND_LIBRARY_PREFIXES ${_old_CMAKE_FIND_LIBRARY_PREFIXES})
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${_old_CMAKE_FIND_LIBRARY_SUFFIXES})
    else()
        find_host_library(libclang_location
            NAMES clang libclang libclang-${clang_version} libclang-${clang_version}.so libclang-${clang_version}.so.1
            PATHS /usr/lib /usr/local/lib /usr/lib/llvm-${clang_version}/lib /usr/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH
            NO_CMAKE_FIND_ROOT_PATH)
    endif()

    if(NOT libclang_location)
        message(WARNING "clang-${clang_version} libclang-${clang_version}-dev are not found (required for ncc naming style check)")
        set(ENABLE_NCC_STYLE OFF)
    else()
        message(STATUS "Found libclang: ${libclang_location}")
    endif()
endif()

# check python requirements_dev.txt
if(ENABLE_NCC_STYLE)
    set(ncc_script_py "${ncc_style_dir}/ncc/ncc.py")

    if(NOT EXISTS ${ncc_script_py})
        message(WARNING "ncc.py is not downloaded via submodule")
        set(ENABLE_NCC_STYLE OFF)
    endif()
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
#                     [SOURCE_DIRECTORIES dir1 dir2 ...]
#                     [STYLE_FILE style_file.style]
#                     [ADDITIONAL_INCLUDE_DIRECTORIES dir1 dir2 ..]
#                     [DEFINITIONS def1 def2 ..])
#
# FOR_TARGET - name of the target
# SOURCE_DIRECTORIES - directory to check sources from
# STYLE_FILE - path to the specific style file
# ADDITIONAL_INCLUDE_DIRECTORIES - additional include directories used in checked headers
# DEFINITIONS - additional definitions passed to preprocessor stage
#
function(ov_ncc_naming_style)
    if(NOT ENABLE_NCC_STYLE)
        return()
    endif()

    cmake_parse_arguments(NCC_STYLE "FAIL"
        "FOR_TARGET;STYLE_FILE" "SOURCE_DIRECTORIES;ADDITIONAL_INCLUDE_DIRECTORIES;DEFINITIONS" ${ARGN})

    foreach(var FOR_TARGET SOURCE_DIRECTORIES)
        if(NOT DEFINED NCC_STYLE_${var})
            message(FATAL_ERROR "${var} is not defined in ov_ncc_naming_style function")
        endif()
    endforeach()

    if(NOT DEFINED NCC_STYLE_STYLE_FILE)
        set(NCC_STYLE_STYLE_FILE ${ncc_style_dir}/openvino.style)
    endif()

    foreach(source_dir IN LISTS NCC_STYLE_SOURCE_DIRECTORIES)
        file(GLOB_RECURSE local_sources "${source_dir}/*.hpp" "${source_dir}/*.cpp")
        list(APPEND sources ${local_sources})
    endforeach()

    list(APPEND NCC_STYLE_ADDITIONAL_INCLUDE_DIRECTORIES ${NCC_STYLE_SOURCE_DIRECTORIES})

    foreach(source_file IN LISTS sources)
        get_filename_component(source_dir "${source_file}" DIRECTORY)
        file(RELATIVE_PATH source_dir_rel "${CMAKE_SOURCE_DIR}" "${source_dir}")
        get_filename_component(source_name "${source_file}" NAME)
        set(output_file "${ncc_style_bin_dir}/${source_dir_rel}/${source_name}.ncc_style")

        add_custom_command(
            OUTPUT
                ${output_file}
            COMMAND
                "${CMAKE_COMMAND}"
                -D "Python3_EXECUTABLE=${Python3_EXECUTABLE}"
                -D "NCC_PY_SCRIPT=${ncc_script_py}"
                -D "INPUT_FILE=${source_file}"
                -D "OUTPUT_FILE=${output_file}"
                -D "DEFINITIONS=${NCC_STYLE_DEFINITIONS}"
                -D "CLANG_LIB_PATH=${libclang_location}"
                -D "STYLE_FILE=${NCC_STYLE_STYLE_FILE}"
                -D "ADDITIONAL_INCLUDE_DIRECTORIES=${NCC_STYLE_ADDITIONAL_INCLUDE_DIRECTORIES}"
                -D "EXPECTED_FAIL=${NCC_STYLE_FAIL}"
                -P "${ncc_style_dir}/ncc_run.cmake"
            DEPENDS
                "${source_file}"
                "${ncc_style_dir}/openvino.style"
                "${ncc_script_py}"
                "${ncc_style_dir}/ncc_run.cmake"
            COMMENT
                "[ncc naming style] ${source_dir_rel}/${source_name}"
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
                        SOURCE_DIRECTORIES "${ncc_style_dir}/self_check"
                        FAIL)
endif()
