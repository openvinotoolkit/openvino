# Copyright (C) 2018-2024  Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_detect_python_module_extension)
    if(NOT ENABLE_PYTHON)
        # python is just disabled
        return()
    endif()

    if(PYTHON_MODULE_EXTENSION)
        # exit if it's already defined
        return()
    endif()

    if(NOT CMAKE_CROSSCOMPILING)
        # in case of native compilation FindPython3.cmake properly detects PYTHON_MODULE_EXTENSION
        return()
    endif()

    if(RISCV64)
        set(python3_config riscv64-linux-gnu-python3-config)
    elseif(AARCH64)
        set(python3_config aarch64-linux-gnu-python3-config)
    elseif(X86_64)
        set(python3_config x86_64-linux-gnu-python3-config)
    else()
        message(WARNING "Python cross-compilation warning: ${OV_ARCH} is unknown for python build. Please, specify PYTHON_MODULE_EXTENSION explicitly")
    endif()

    find_host_program(python3_config_exec NAMES ${python3_config})
    if(python3_config_exec)
        execute_process(COMMAND ${python3_config_exec} --extension-suffix
                        RESULT_VARIABLE EXIT_CODE
                        OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
                        ERROR_VARIABLE ERROR_TEXT
                        OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT EXIT_CODE EQUAL 0)
            message(FATAL_ERROR "Internal error: failed to execute ${python3_config_exec}")
        endif()
        set(PYTHON_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION} PARENT_SCOPE)
    else()
        message(FATAL_ERROR [=[PYTHON_MODULE_EXTENSION will not be properly detected. Please, either:
    1. Install libpython3-dev for target architecture
    2. Explicitly specify PYTHON_MODULE_EXTENSION
    ]=])
    endif()
endfunction()

# Wrapper for find_package(Python3) to allow cross-compilation
macro(ov_find_python3 find_package_mode)
    # Settings for FindPython3.cmake
    if(NOT DEFINED Python3_USE_STATIC_LIBS)
        set(Python3_USE_STATIC_LIBS OFF)
    endif()

    if(NOT DEFINED Python3_FIND_VIRTUALENV)
        set(Python3_FIND_VIRTUALENV FIRST)
    endif()

    if(NOT DEFINED Python3_FIND_IMPLEMENTATIONS)
        set(Python3_FIND_IMPLEMENTATIONS CPython PyPy)
    endif()

    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set(python3_development_component Development.Module)
    else()
        set(python3_development_component Development)
    endif()

    if(CMAKE_CROSSCOMPILING AND LINUX)
        # allow to find python headers from host in case of cross-compilation
        # e.g. install libpython3-dev:<foreign arch> and finds its headers
        set(_old_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
        ov_cross_compile_define_debian_arch()
    endif()

    find_package(Python3 ${find_package_mode} COMPONENTS Interpreter ${python3_development_component})

    if(CMAKE_CROSSCOMPILING AND LINUX)
        ov_cross_compile_define_debian_arch_reset()
        set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ${_old_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})
    endif()

    unset(python3_development_component)
endmacro()
