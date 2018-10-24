# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# FindIE
# ------
#
#   You can specify the path to Inference Engine files in IE_ROOT_DIR
#
# This will define the following variables:
#
#   InferenceEngine_FOUND        - True if the system has the Inference Engine library
#   InferenceEngine_INCLUDE_DIRS - Inference Engine include directories
#   InferenceEngine_LIBRARIES    - Inference Engine libraries
#
# and the following imported targets:
#
#   IE::inference_engine    - The Inference Engine library
#


set(InferenceEngine_FOUND FALSE)

if(TARGET IE::inference_engine)
    set(InferenceEngine_FOUND TRUE)
    get_target_property(InferenceEngine_INCLUDE_DIRS IE::inference_engine INTERFACE_INCLUDE_DIRECTORIES)
    set(InferenceEngine_LIBRARIES IE::inference_engine)
else()
    if (WIN32)
        set(_ARCH intel64)
    else()
        if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
            set(_ARCH intel64)
        elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i386")
            set(_ARCH ia32)
        endif()
    endif()

    # check whether setvars.sh is sourced
    if(NOT IE_ROOT_DIR AND (DEFINED ENV{InferenceEngine_DIR} OR InferenceEngine_DIR OR DEFINED ENV{INTEL_CVSDK_DIR}))
        if (EXISTS "${InferenceEngine_DIR}")
            # InferenceEngine_DIR manually set via command line params
            set(IE_ROOT_DIR "${InferenceEngine_DIR}/..")
        elseif (EXISTS "$ENV{InferenceEngine_DIR}")
            # InferenceEngine_DIR manually set via env
            set(IE_ROOT_DIR "$ENV{InferenceEngine_DIR}/..")
        elseif (EXISTS "$ENV{INTEL_CVSDK_DIR}/inference_engine")
            # if we installed DL SDK
            set(IE_ROOT_DIR "$ENV{INTEL_CVSDK_DIR}/inference_engine")
        elseif (EXISTS "$ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine")
            # CV SDK is installed
            set(IE_ROOT_DIR "$ENV{INTEL_CVSDK_DIR}/deployment_tools/inference_engine")
        endif()
    endif()

    if(IE_ROOT_DIR)
        if (WIN32)
            set(_OS_PATH "")
        else()
           if (NOT EXISTS "/etc/lsb-release")
                execute_process(COMMAND find /usr/lib/ -maxdepth 1 -type f -name *-release -exec cat {} \;
                            OUTPUT_VARIABLE release_data RESULT_VARIABLE result)
                set(name_regex "NAME=\"([^ \"\n]*).*\"\n")
                set(version_regex "VERSION=\"([0-9]+(\\.[0-9]+)?)[^\n]*\"")
            else()
                #linux version detection using cat /etc/lsb-release
                file(READ "/etc/lsb-release" release_data)
                set(name_regex "DISTRIB_ID=([^ \n]*)\n")
                set(version_regex "DISTRIB_RELEASE=([0-9]+(\\.[0-9]+)?)")
            endif()

            string(REGEX MATCH ${name_regex} name ${release_data})
            set(os_name ${CMAKE_MATCH_1})

            string(REGEX MATCH ${version_regex} version ${release_data})
            set(os_name "${os_name} ${CMAKE_MATCH_1}")

            if (NOT os_name)
                message(FATAL_ERROR "Cannot detect OS via reading /etc/*-release:\n ${release_data}")
            endif()

            message (STATUS "/etc/*-release distrib: ${os_name}")

            if (${os_name} STREQUAL "Ubuntu 14.04")
                set(_OS_PATH "ubuntu_14.04/")
            elseif (${os_name} STREQUAL "Ubuntu 16.04")
                set(_OS_PATH "ubuntu_16.04/")
            elseif (${os_name} STREQUAL "CentOS 7")
                set(_OS_PATH "centos_7.4/")
            elseif (${os_name} STREQUAL "poky 2.0")
                set(_OS_PATH "ubuntu_16.04/")
            else()
                message(FATAL_ERROR "${os_name} is not supported. List of supported OS: Ubuntu 14.04, Ubuntu 16.04, CentOS 7")
            endif()
        endif()
    endif()

    if(IE_INCLUDE_DIR AND NOT "${IE_ROOT_DIR}/include" EQUAL "${IE_INCLUDE_DIR}")
        unset(IE_INCLUDE_DIR CACHE)
    endif()

    if(IE_LIBRARY AND NOT "${IE_ROOT_DIR}/lib/${_OS_PATH}/${_ARCH}" EQUAL "${IE_LIBRARY}")
        unset(IE_LIBRARY CACHE)
    endif()

    set(_IE_ROOT_INCLUDE_DIR "${IE_ROOT_DIR}/include")
    set(_IE_ROOT_LIBRARY "${IE_ROOT_DIR}/lib/${_OS_PATH}/${_ARCH}")
    

    find_path(IE_INCLUDE_DIR inference_engine.hpp "${_IE_ROOT_INCLUDE_DIR}")
    #message("InferenceEngine_INCLUDE_DIR=${IE_INCLUDE_DIR}:${_IE_ROOT_INCLUDE_DIR}")

    include(FindPackageHandleStandardArgs)
    if (WIN32)
        find_library(IE_RELEASE_LIBRARY inference_engine "${_IE_ROOT_LIBRARY}/Release")
        find_library(IE_DEBUG_LIBRARY inference_engine "${_IE_ROOT_LIBRARY}/Debug")
        find_package_handle_standard_args(  IE
                                            REQUIRED_VARS IE_RELEASE_LIBRARY IE_DEBUG_LIBRARY IE_INCLUDE_DIR
                                            FAIL_MESSAGE "Inference Engine cannot be found at ${_IE_ROOT_LIBRARY}. Please consult InferenceEgnineConfig.cmake module's help page.")
    else()
        find_library(IE_LIBRARY inference_engine "${_IE_ROOT_LIBRARY}")
        find_package_handle_standard_args(  IE
                                            REQUIRED_VARS IE_LIBRARY IE_INCLUDE_DIR
                                            FAIL_MESSAGE "Inference Engine cannot be found at ${_IE_ROOT_LIBRARY}. Please consult InferenceEgnineConfig.cmake module's help page.")
    endif()
    if(IE_FOUND)
        add_library(IE::inference_engine SHARED IMPORTED GLOBAL)

        if (WIN32)
            set_property(TARGET IE::inference_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
            set_property(TARGET IE::inference_engine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

            set_target_properties(IE::inference_engine PROPERTIES
                    IMPORTED_IMPLIB_RELEASE    "${IE_RELEASE_LIBRARY}"
                    IMPORTED_IMPLIB_DEBUG      "${IE_DEBUG_LIBRARY}"
                    MAP_IMPORTED_CONFIG_DEBUG Debug
                    MAP_IMPORTED_CONFIG_RELEASE Release
                    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
                    INTERFACE_INCLUDE_DIRECTORIES "${IE_INCLUDE_DIR}")
        else()
            set_target_properties(IE::inference_engine PROPERTIES
                    IMPORTED_LOCATION "${IE_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${IE_INCLUDE_DIR}")
            target_link_libraries(IE::inference_engine INTERFACE ${CMAKE_DL_LIBS})
        endif()

        set(InferenceEngine_INCLUDE_DIRS ${IE_INCLUDE_DIR})
        set(InferenceEngine_LIBRARIES IE::inference_engine)
        set(InferenceEngine_FOUND TRUE)
    endif()
endif()

