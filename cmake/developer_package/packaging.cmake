# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CPackComponent)
unset(IE_CPACK_COMPONENTS_ALL CACHE)

#
# ie_cpack_set_library_dir()
#
# Set library directory for cpack
#
function(ie_cpack_set_library_dir)
    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH runtime/lib PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH runtime/bin PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH runtime/lib PARENT_SCOPE)
    else()
        set(IE_CPACK_LIBRARY_PATH runtime/lib PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH runtime/lib PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH runtime/lib PARENT_SCOPE)
    endif()
endfunction()

ie_cpack_set_library_dir()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
macro(ie_cpack_add_component NAME)
    list(APPEND IE_CPACK_COMPONENTS_ALL ${NAME})
    set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)
    cpack_add_component(${NAME} ${ARGN})
endmacro()

macro(ie_cpack)
    set(CPACK_GENERATOR "TGZ")
    string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "${CI_BUILD_NUMBER}")
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
    endif()
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_PACKAGE_VENDOR "Intel")
    set(CPACK_COMPONENTS_ALL ${ARGN})
    set(CPACK_STRIP_FILES ON)

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    include(CPack)
endmacro()
