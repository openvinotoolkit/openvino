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
        # enable dependencies between components
        set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
        # homepage
        set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://docs.openvino.ai/")
        # enable for debug cpack run
        if(NOT DEFINED CPACK_DEBIAN_PACKAGE_DEBUG)
            set(CPACK_DEBIAN_PACKAGE_DEBUG OFF)
        endif()

        # automatic dependencies discovering
        set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS ON)
        # OpenVINO does not have backward and forward compatibility
        set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS_POLICY "=")

        # TODO
        # add preinst script for core to ask about aptin

        # ngraph
        set(CPACK_COMPONENT_NGRAPH_DESCRIPTION "OpenVINO ngraph")
        # admin, devel, doc, see https://www.debian.org/doc/debian-policy/ch-archive.html#s-subsections
        # set(CPACK_DEBIAN_ngraph_PACKAGE_SECTION devel)
        # see required, important, standard, optional, extra
        # https://www.debian.org/doc/debian-policy/ch-archive.html#s-priorities
        # set(CPACK_DEBIAN_ngraph_PACKAGE_PRIORITY standard)

        # ngraph-dev
        set(CPACK_COMPONENT_NGRAPH_DEV_DESCRIPTION "OpenVINO ngraph development files")
        set(CPACK_COMPONENT_NGRAPH_DEV_DEPENDS "ngraph")

        # core
        set(CPACK_COMPONENT_CORE_DESCRIPTION "OpenVINO core runtime")
        set(CPACK_COMPONENT_CORE_DEPENDS "ngraph")
        # TODO: should be discovered automatically
        # TODO: install build dependencies libpugixml-dev, libtbb-dev
        set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libpugixml1v5")

        # hetero
        set(CPACK_COMPONENT_hetero_DESCRIPTION "OpenVINO Hetero plugin")
        set(CPACK_COMPONENT_HETERO_DEPENDS "core")

        # core_dev
        set(CPACK_COMPONENT_CODE_DEV_DESCRIPTION "OpenVINO core dev runtime")
        set(CPACK_COMPONENT_COR_DEV_DEPENDS "ngraph_dev;core")
        # set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libtbb-dev")
        # set(CPACK_DEBIAN_core_dev_PACKAGE_CONFLICTS "!!!")

        # core_c
        set(CPACK_COMPONENT_core_c_DESCRIPTION "OpenVINO C core runtime")
        set(CPACK_COMPONENT_CORE_C_DEPENDS "core")

        # core_c_dev
        set(CPACK_COMPONENT_core_c_dev_DESCRIPTION "OpenVINO C dev runtime")
        set(CPACK_COMPONENT_CORE_C_DEV_DEPENDS "core_c;core_dev")
        # set(CPACK_DEBIAN_core_dev_PACKAGE_CONFLICTS "!!!")

        # core tools
        set(CPACK_COMPONENT_CORE_TOOLS_DESCRIPTION "OpenVINO Core Tools")
        set(CPACK_COMPONENT_CORE_TOOLS_DEPENDS "core")
        set(CPACK_DEBIAN_CORE_TOOLS_PACKAGE_RECOMMENDS "openvino-hetero (= ${CPACK_PACKAGE_VERSION})")

        # cpp_samples
        set(CPACK_COMPONENT_CPP_SAMPLES_DESCRIPTION "OpenVINO C++ samples")
        set(CPACK_COMPONENT_CPP_SAMPLES_DEPENDS "core_dev")
        set(CPACK_DEBIAN_CPP_SAMPLES_PACKAGE_RECOMMENDS "openvino-hetero (= ${CPACK_PACKAGE_VERSION})")
        set(CPACK_DEBIAN_CPP_SAMPLES_PACKAGE_DEPENDS "libgflags-dev, nlohmann-json3-dev, zlib1g-dev")

        # c_samples
        set(CPACK_COMPONENT_C_SAMPLES_DESCRIPTION "OpenVINO C samples")
        set(CPACK_COMPONENT_C_SAMPLES_DEPENDS "core_c_dev")
    endif()

    include(CPack)
endmacro()
