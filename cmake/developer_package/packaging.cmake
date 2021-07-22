# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)
include(CPackComponent)

#
# ie_cpack_set_library_dir()
#
# Set library directory for cpack
#
set(IE_CPACK_IE_DIR       deployment_tools/inference_engine)
function(ie_cpack_set_library_dir)
    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH ${IE_CPACK_IE_DIR}/bin/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
    else()
        set(IE_CPACK_LIBRARY_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER} PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER} PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH ${IE_CPACK_IE_DIR}/lib/${ARCH_FOLDER} PARENT_SCOPE)
    endif()
endfunction()

ie_cpack_set_library_dir()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
unset(IE_CPACK_COMPONENTS_ALL CACHE)
macro(ie_cpack_add_component NAME)
    list(APPEND IE_CPACK_COMPONENTS_ALL ${NAME})
    set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)

    cpack_add_component(${NAME} ${args})
endmacro()

# create test component
if(ENABLE_TESTS)
    cpack_add_component(tests DISABLED)
endif()

macro(ie_cpack)
    set(CPACK_GENERATOR "TGZ")
    set(CPACK_SOURCE_GENERATOR "")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenVINO toolkit")
    set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED OFF)
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON) # multiple components
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")
    set(CPACK_VERBATIM_VARIABLES ON)
    set(CPACK_COMPONENTS_ALL ${ARGN})
    if (NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()
    set(CPACK_THREADS 8)

    string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "${CI_BUILD_NUMBER}")
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
    endif()

    foreach(ver IN LISTS MAJOR MINOR PATCH)
        if(DEFINED IE_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${IE_VERSION_${ver}})
        endif()
    endforeach()

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    include(CPack)
endmacro()
