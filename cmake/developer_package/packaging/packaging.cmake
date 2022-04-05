# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CMakeParseArguments)
include(CPackComponent)
include(GNUInstallDirs)

#
# ov_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_cpack_set_dirs)
    # common IRC package locations
    # TODO: move current variables to OpenVINO specific locations
    set(OV_CPACK_INCLUDEDIR runtime/include)
    set(OV_CPACK_IE_CMAKEDIR runtime/cmake)
    set(OV_CPACK_NGRAPH_CMAKEDIR runtime/cmake)
    set(OV_CPACK_OPENVINO_CMAKEDIR runtime/cmake)
    set(OV_CPACK_DOCDIR docs)
    set(OV_CPACK_SAMPLESDIR samples)
    set(OV_CPACK_PYTHONDIR python)
    set(OV_CPACK_WHEELSDIR tools)
    set(OV_CPACK_TOOLSDIR tools)
    set(OV_CPACK_DEVREQDIR tools)

    if(WIN32)
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_RUNTIMEDIR runtime/bin/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_PLUGINSDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
    else()
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_RUNTIMEDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_PLUGINSDIR runtime/lib/${ARCH_FOLDER})
    endif()

    # for BW compatibility
    set(IE_CPACK_LIBRARY_PATH ${OV_CPACK_LIBRARYDIR})
    set(IE_CPACK_RUNTIME_PATH ${OV_CPACK_RUNTIMEDIR})
    set(IE_CPACK_ARCHIVE_PATH ${OV_CPACK_ARCHIVEDIR})
endmacro()

ov_cpack_set_dirs()

#
# ie_cpack_add_component(NAME ...)
#
# Wraps original `cpack_add_component` and adds component to internal IE list
#
unset(IE_CPACK_COMPONENTS_ALL CACHE)
function(ie_cpack_add_component name)
    if(NOT ${name} IN_LIST IE_CPACK_COMPONENTS_ALL)
        cpack_add_component(${name} ${args})

        list(APPEND IE_CPACK_COMPONENTS_ALL ${name})
        set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE STRING "" FORCE)
    endif()
endfunction()

# create `tests` component
if(ENABLE_TESTS)
    cpack_add_component(tests DISABLED)
endif()

#
# List of public OpenVINO components
#

# core components
set(OV_CPACK_COMP_CORE "core")
set(OV_CPACK_COMP_CORE_C "core_c")
set(OV_CPACK_COMP_CORE_DEV "core_dev")
set(OV_CPACK_COMP_CORE_C_DEV "core_c_dev")
# licensing
set(OV_CPACK_COMP_LICENSING "licensing")
# samples
set(OV_CPACK_COMP_CPP_SAMPLES "cpp_samples")
set(OV_CPACK_COMP_C_SAMPLES "c_samples")
set(OV_CPACK_COMP_PYTHON_SAMPLES "python_samples")
# python
set(OV_CPACK_COMP_PYTHON_IE_API "pyie")
set(OV_CPACK_COMP_PYTHON_NGRAPH "pyngraph")
set(OV_CPACK_COMP_PYTHON_OPENVINO "pyopenvino")
# tools
set(OV_CPACK_COMP_CORE_TOOLS "core_tools")
set(OV_CPACK_COMP_DEV_REQ_FILES "openvino_dev_req_files")
set(OV_CPACK_COMP_DEPLOYMENT_MANAGER "deployment_manager")

# Include Debian specific configuration file:
# - overrides directories set by ov_cpack_set_dirs()
# - merges some components
if(CPACK_GENERATOR STREQUAL "DEB")
    include(packaging/debian)
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
    set(CPACK_PACKAGE_CONTACT "OpenVINO Developers <openvino@intel.com>")
    set(CPACK_VERBATIM_VARIABLES ON)
    set(CPACK_COMPONENTS_ALL ${ARGN})

    # TODO: check whether we need it
    # default permissions for directories creation
    set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE)

    # archive operations can be run in parallels since CMake 3.20
    set(CPACK_THREADS 8)

    if(NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()

    if(WIN32)
        set(CPACK_PACKAGE_NAME openvino_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME openvino)
    endif()

    set(CPACK_PACKAGE_VERSION "${OpenVINO_VERSION}")
    foreach(ver MAJOR MINOR PATCH)
        if(DEFINED OpenVINO_VERSION_${ver})
            set(CPACK_PACKAGE_VERSION_${ver} ${OpenVINO_VERSION_${ver}})
        else()
            message(FATAL_ERROR "Internal: OpenVINO_VERSION_${ver} variable is not defined")
        endif()
    endforeach()

    if(OS_FOLDER)
        set(CPACK_SYSTEM_NAME "${OS_FOLDER}")
    endif()

    # generator specific variables
    if(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|ZIP)$")
        # multiple packages are generated
        set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    elseif(CPACK_GENERATOR STREQUAL "DEB")
        # include Debian dedicated per-component configuration file
        # NOTE: private modules need to define ov_debian_components macro
        # for custom debian packages configuration
        if(COMMAND ov_debian_components)
            ov_debian_components()
        endif()
    endif()

    include(CPack)
endmacro()
