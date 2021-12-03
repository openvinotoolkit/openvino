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
function(ie_cpack_set_library_dir)
    if(WIN32)
        set(IE_CPACK_LIBRARY_PATH lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH bin/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH lib/${ARCH_FOLDER}/$<CONFIG> PARENT_SCOPE)
    else()
        set(IE_CPACK_LIBRARY_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        set(IE_CPACK_RUNTIME_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
        set(IE_CPACK_ARCHIVE_PATH lib/${ARCH_FOLDER} PARENT_SCOPE)
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
    if(NOT DEFINED CPACK_GENERATOR)
        set(CPACK_GENERATOR "TGZ")
    endif()
    set(CPACK_SOURCE_GENERATOR "") # not used
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenVINO™ Toolkit")
    set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED OFF)
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    set(CPACK_PACKAGE_VENDOR "Intel Corporation")
    set(CPACK_PACKAGE_CONTACT "openvino@intel.com")
    set(CPACK_VERBATIM_VARIABLES ON)
    set(CPACK_COMPONENTS_ALL ${ARGN})

    # archive operations can be run in parallels since CMake 3.20
    set(CPACK_THREADS 8)

    if (NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()

    string(REPLACE "/" "_" CPACK_PACKAGE_VERSION "2022.1.1")
    if(WIN32)
        set(CPACK_PACKAGE_NAME openvino_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME openvino)
    endif()

    foreach(ver IN LISTS MAJOR MINOR PATCH)
        if(DEFINED IE_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${IE_VERSION_${ver}})
        endif()
    endforeach()

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    # generator specific variables
    if(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|ZIP)$")
        message(FATAL_ERROR "CPACK_GENERATORCPACK_GENERATORCPACK_GENERATOR")
        # multiple packages are generated
        set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    elseif(CPACK_GENERATOR STREQUAL "DEB")
        # per component configuration
        # list of components:
        # - ngraph
        # - core
        # - core_c
        # - ngraph_dev
        # - core_dev
        # - core_c_dev
        # - licensing
        # - docs
        # - core_tools
        # - [python .*]
        # - (cpu|gpu|hetero)

        # multiple packages are generated
        set(CPACK_DEB_COMPONENT_INSTALL ON)
        # automatically find dependencies for binaries
        set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

        # ngraph
        set(CPACK_DEBIAN_ngraph_PACKAGE_NAME "openvino-ngraph")
    endif()

    include(CPack)
endmacro()
