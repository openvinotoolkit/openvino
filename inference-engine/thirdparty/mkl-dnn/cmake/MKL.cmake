#===============================================================================
# Copyright 2016-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

# Locate Intel(R) MKL installation using MKLROOT or look in
# ${CMAKE_CURRENT_SOURCE_DIR}/external
#===============================================================================

if(MKL_cmake_included)
    return()
endif()
set(MKL_cmake_included true)

function(detect_mkl LIBNAME)
    if(HAVE_MKL)
        return()
    endif()

    message(STATUS "Detecting Intel(R) MKL: trying ${LIBNAME}")

    find_path(MKLINC mkl_cblas.h
        HINTS ${MKLROOT}/include $ENV{MKLROOT}/include)
    if(NOT MKLINC)
        file(GLOB_RECURSE MKLINC
                ${CMAKE_CURRENT_SOURCE_DIR}/external/*/mkl_cblas.h)
        if(MKLINC)
            # if user has multiple version under external/ then guess last
            # one alphabetically is "latest" and warn
            list(LENGTH MKLINC MKLINCLEN)
            if(MKLINCLEN GREATER 1)
                list(SORT MKLINC)
                list(REVERSE MKLINC)
                # message(STATUS "MKLINC found ${MKLINCLEN} files:")
                # foreach(LOCN IN LISTS MKLINC)
                #     message(STATUS "       ${LOCN}")
                # endforeach()
                list(GET MKLINC 0 MKLINCLST)
                set(MKLINC "${MKLINCLST}")
                # message(WARNING "MKLINC guessing... ${MKLINC}.  "
                #     "Please check that above dir has the desired mkl_cblas.h")
            endif()
            get_filename_component(MKLINC ${MKLINC} PATH)
        endif()
    endif()
    if(NOT MKLINC)
        return()
    endif()

    get_filename_component(__mklinc_root "${MKLINC}" PATH)
    find_library(MKLLIB NAMES ${LIBNAME}
        HINTS   ${MKLROOT}/lib ${MKLROOT}/lib/intel64
                $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64
                ${__mklinc_root}/lib ${__mklinc_root}/lib/intel64)
    if(NOT MKLLIB)
        return()
    endif()

    if(WIN32)
        set(MKLREDIST ${MKLINC}/../../redist/)
        find_file(MKLDLL NAMES ${LIBNAME}.dll
            HINTS
                ${MKLREDIST}/mkl
                ${MKLREDIST}/intel64/mkl
                ${__mklinc_root}/lib)
        if(NOT MKLDLL)
            return()
        endif()
    endif()

    if(UNIX AND LIBNAME MATCHES "mklml.*")
        # Although MKL-ML depends on shared object functions such as dlopen and
        # dladdr it is not linked against libdl. This causes link failures when
        # MKL-DNN is build with the gold linker (e.g. -fuse-ld=gold).
        list(APPEND EXTRA_LIBS dl)
        set(EXTRA_LIBS "${EXTRA_LIBS}" PARENT_SCOPE)
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        get_filename_component(MKLLIBPATH ${MKLLIB} PATH)
        find_library(MKLIOMP5LIB
            NAMES "iomp5" "iomp5md" "libiomp5" "libiomp5md"
            HINTS   ${MKLLIBPATH}
                    ${MKLLIBPATH}/../../lib
                    ${MKLLIBPATH}/../../../lib/intel64
                    ${MKLLIBPATH}/../../compiler/lib
                    ${MKLLIBPATH}/../../../compiler/lib/intel64)
        if(NOT MKLIOMP5LIB)
            return()
        endif()
        if(WIN32)
            find_file(MKLIOMP5DLL
                NAMES "libiomp5.dll" "libiomp5md.dll"
                HINTS ${MKLREDIST}/../compiler ${__mklinc_root}/lib)
            if(NOT MKLIOMP5DLL)
                return()
            endif()
        endif()
    else()
        set(MKLIOMP5LIB)
        set(MKLIOMP5DLL)
    endif()

    get_filename_component(MKLLIBPATH "${MKLLIB}" PATH)
    string(FIND "${MKLLIBPATH}" ${CMAKE_CURRENT_SOURCE_DIR}/external __idx)
    if(${__idx} EQUAL 0)
        if(WIN32)
            if(MINGW)
                # We need to install *.dll into bin/ instead of lib/.
                install(PROGRAMS ${MKLDLL} DESTINATION bin)
            else()
                install(PROGRAMS ${MKLDLL} DESTINATION lib)
            endif()
        else()
            install(PROGRAMS ${MKLLIB} DESTINATION lib)
        endif()
        if(MKLIOMP5LIB)
            if(WIN32)
                if(MINGW)
                    # We need to install *.dll into bin/ instead of lib/.
                    install(PROGRAMS ${MKLIOMP5DLL} DESTINATION bin)
                else()
                    install(PROGRAMS ${MKLIOMP5DLL} DESTINATION lib)
                endif()
            else()
                install(PROGRAMS ${MKLIOMP5LIB} DESTINATION lib)
            endif()
        endif()
    endif()

    if(WIN32)
        # Add paths to DLL to %PATH% on Windows
        get_filename_component(MKLDLLPATH "${MKLDLL}" PATH)
        set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}\;${MKLDLLPATH}")
        set(CTESTCONFIG_PATH "${CTESTCONFIG_PATH}" PARENT_SCOPE)
    endif()

    # TODO: cache the value
    set(HAVE_MKL TRUE PARENT_SCOPE)
    set(MKLINC ${MKLINC} PARENT_SCOPE)
    set(MKLLIB "${MKLLIB}" PARENT_SCOPE)

    if(WIN32)
        set(MKLDLL "${MKLDLL}" PARENT_SCOPE)
    endif()
    if(MKLIOMP5LIB)
        set(MKLIOMP5LIB "${MKLIOMP5LIB}" PARENT_SCOPE)
    endif()
    if(WIN32 AND MKLIOMP5DLL)
        set(MKLIOMP5DLL "${MKLIOMP5DLL}" PARENT_SCOPE)
    endif()
endfunction()

detect_mkl("mklml_intel")
detect_mkl("mklml")
detect_mkl("mkl_rt")

if(HAVE_MKL)
    add_definitions(-DUSE_MKL -DUSE_CBLAS)
    include_directories(AFTER ${MKLINC})
    list(APPEND mkldnn_LINKER_LIBS ${MKLLIB})

    set(MSG "Intel(R) MKL:")
    message(STATUS "${MSG} include ${MKLINC}")
    message(STATUS "${MSG} lib ${MKLLIB}")
    if(MKLIOMP5LIB)
        message(STATUS "${MSG} OpenMP lib ${MKLIOMP5LIB}")
    else()
        message(STATUS "${MSG} OpenMP lib provided by compiler")
    endif()
    if(WIN32)
        message(STATUS "${MSG} dll ${MKLDLL}")
        if(MKLIOMP5DLL)
            message(STATUS "${MSG} OpenMP dll ${MKLIOMP5DLL}")
        else()
            message(STATUS "${MSG} OpenMP dll provided by compiler")
        endif()
    endif()
else()
    if(DEFINED ENV{FAIL_WITHOUT_MKL} OR DEFINED FAIL_WITHOUT_MKL)
        set(SEVERITY "FATAL_ERROR")
    else()
        set(SEVERITY "WARNING")
    endif()
    message(${SEVERITY}
        "Intel(R) MKL not found. Some performance features may not be "
        "available. Please run scripts/prepare_mkl.sh to download a minimal "
        "set of libraries or get a full version from "
        "https://software.intel.com/en-us/intel-mkl")
endif()
