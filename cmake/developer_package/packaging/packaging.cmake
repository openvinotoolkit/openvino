# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CPackComponent)

#
# ov_get_pyversion()
#
function(ov_get_pyversion pyversion)
    find_package(PythonInterp 3 QUIET)
    if(PYTHONINTERP_FOUND)
        set(${pyversion} "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}" PARENT_SCOPE)
    else()
        set(${pyversion} "NOT-FOUND" PARENT_SCOPE)
    endif()
endfunction()

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
    set(OV_CPACK_WHEELSDIR tools)
    set(OV_CPACK_TOOLSDIR tools)
    set(OV_CPACK_DEVREQDIR tools)

    ov_get_pyversion(pyversion)
    if(pyversion)
        set(OV_CPACK_PYTHONDIR python/${pyversion})
    endif()

    if(WIN32)
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_RUNTIMEDIR runtime/bin/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_WHEEL_RUNTIMEDIR runtime/bin/${ARCH_FOLDER}/Release)
    elseif(APPLE)
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_RUNTIMEDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER}/$<CONFIG>)
        set(OV_WHEEL_RUNTIMEDIR runtime/lib/${ARCH_FOLDER}/Release)
    else()
        set(OV_CPACK_LIBRARYDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_RUNTIMEDIR runtime/lib/${ARCH_FOLDER})
        set(OV_CPACK_ARCHIVEDIR runtime/lib/${ARCH_FOLDER})
        set(OV_WHEEL_RUNTIMEDIR ${OV_CPACK_RUNTIMEDIR})
    endif()
    set(OV_CPACK_PLUGINSDIR ${OV_CPACK_RUNTIMEDIR})

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
function(ie_cpack_add_component name)
    if(NOT ${name} IN_LIST IE_CPACK_COMPONENTS_ALL)
        cpack_add_component(${name} ${ARGN})

        # need to store informarion about cpack_add_component arguments in CMakeCache.txt
        # to restore it later
        set(_${name}_cpack_component_args "${ARGN}" CACHE INTERNAL "Argument for cpack_add_component for ${name} cpack component" FORCE)

        list(APPEND IE_CPACK_COMPONENTS_ALL ${name})
        set(IE_CPACK_COMPONENTS_ALL "${IE_CPACK_COMPONENTS_ALL}" CACHE INTERNAL "" FORCE)
    endif()
endfunction()

foreach(comp IN LISTS IE_CPACK_COMPONENTS_ALL)
    unset(_${comp}_cpack_component_args)
endforeach()
unset(IE_CPACK_COMPONENTS_ALL CACHE)

# create `tests` component
if(ENABLE_TESTS)
    cpack_add_component(tests HIDDEN)
endif()

#
#  ov_install_with_name(<FILE> <COMPONENT>)
#
# if <FILE> is a symlink, we resolve it, but install file with a name of symlink
#
function(ov_install_with_name file component)
    if((APPLE AND file MATCHES "^[^\.]+\.[0-9]+${CMAKE_SHARED_LIBRARY_SUFFIX}$") OR
                (file MATCHES "^.*\.${CMAKE_SHARED_LIBRARY_SUFFIX}\.[0-9]+$"))
        if(IS_SYMLINK "${file}")
            get_filename_component(actual_name "${file}" NAME)
            get_filename_component(file "${file}" REALPATH)
            set(install_rename RENAME "${actual_name}")
        endif()

        install(FILES "${file}"
                DESTINATION runtime/3rdparty/${component}/lib
                COMPONENT ${component}
                EXCLUDE_FROM_ALL
                ${install_rename})

        set("${component}_INSTALLED" ON PARENT_SCOPE)
    endif()
endfunction()

#
# List of public OpenVINO components
#

macro(ov_define_component_names)
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
    set(OV_CPACK_COMP_PYTHON_WHEELS "python_wheels")
    # tools
    set(OV_CPACK_COMP_CORE_TOOLS "core_tools")
    set(OV_CPACK_COMP_DEV_REQ_FILES "openvino_dev_req_files")
    set(OV_CPACK_COMP_DEPLOYMENT_MANAGER "deployment_manager")
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES "install_dependencies")
    set(OV_CPACK_COMP_SETUPVARS "setupvars")
endmacro()

ov_define_component_names()

# Include Debian specific configuration file:
# - overrides directories set by ov_debian_cpack_set_dirs()
# - merges some components using ov_override_component_names()
# - sets ov_debian_specific_settings() with DEB generator variables
# - defines the following helper functions:
#  - ov_add_lintian_suppression()
#  - ov_add_latest_component()
if(CPACK_GENERATOR STREQUAL "DEB")
    include(packaging/debian/debian)
elseif(CPACK_GENERATOR STREQUAL "RPM")
    include(packaging/rpm/rpm)
elseif(CPACK_GENERATOR STREQUAL "NSIS")
    include(packaging/nsis)
elseif(CPACK_GENERATOR MATCHES "^(CONDA-FORGE|BREW)$")
    include(packaging/common-libraries)
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
    # TODO: set proper license file for Windows installer
    set(CPACK_RESOURCE_FILE_LICENSE "${OpenVINO_SOURCE_DIR}/LICENSE")

    # default permissions for directories creation
    set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE OWNER_EXECUTE
        WORLD_READ WORLD_EXECUTE)

    # archive operations can be run in parallel since CMake 3.20
    set(CPACK_THREADS 8)

    if(NOT DEFINED CPACK_STRIP_FILES)
        set(CPACK_STRIP_FILES ON)
    endif()

    # TODO: replace with openvino and handle multi-config generators case
    if(WIN32)
        set(CPACK_PACKAGE_NAME inference-engine_${CMAKE_BUILD_TYPE})
    else()
        set(CPACK_PACKAGE_NAME inference-engine)
    endif()

    set(CPACK_PACKAGE_VERSION "${OpenVINO_VERSION}")
    # build version can be empty in case we are running cmake out of git repository
    if(NOT OpenVINO_VERSION_BUILD STREQUAL "000")
        set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION}.${OpenVINO_VERSION_BUILD}")
    endif()

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

    # include GENERATOR dedicated per-component configuration file
    # NOTE: private modules need to define ov_cpack_settings macro
    # for custom  packages configuration
    if(COMMAND ov_cpack_settings)
        ov_cpack_settings()
    endif()

    # generator specific variables
    if(CPACK_GENERATOR MATCHES "^(7Z|TBZ2|TGZ|TXZ|TZ|ZIP)$")
        # New in version 3.18
        set(CPACK_ARCHIVE_THREADS 8)
        # multiple packages are generated
        set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    endif()

    include(CPack)
endmacro()
