# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

#
# ov_common_libraries_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_common_libraries_cpack_set_dirs)
    # override default locations for common libraries
    set(OV_CPACK_TOOLSDIR ${CMAKE_INSTALL_BINDIR}) # only C++ tools are here
    set(OV_CPACK_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    set(OV_CPACK_LIBRARYDIR ${CMAKE_INSTALL_LIBDIR})
    if(WIN32)
        set(OV_CPACK_RUNTIMEDIR ${CMAKE_INSTALL_BINDIR})
    else()
        set(OV_CPACK_RUNTIMEDIR ${CMAKE_INSTALL_LIBDIR})
    endif()
    set(OV_CPACK_ARCHIVEDIR ${CMAKE_INSTALL_LIBDIR})
    if(CPACK_GENERATOR MATCHES "^(CONAN|VCPKG)$")
        set(OV_CPACK_IE_CMAKEDIR ${CMAKE_INSTALL_DATADIR}/openvino)
        set(OV_CPACK_NGRAPH_CMAKEDIR ${CMAKE_INSTALL_DATADIR}/openvino)
        set(OV_CPACK_OPENVINO_CMAKEDIR ${CMAKE_INSTALL_DATADIR}/openvino)
        set(OV_CPACK_PLUGINSDIR ${OV_CPACK_RUNTIMEDIR})
    else()
        set(OV_CPACK_IE_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/inferenceengine${OpenVINO_VERSION})
        set(OV_CPACK_NGRAPH_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ngraph${OpenVINO_VERSION})
        set(OV_CPACK_OPENVINO_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/openvino${OpenVINO_VERSION})
        set(OV_CPACK_PLUGINSDIR ${OV_CPACK_RUNTIMEDIR}/openvino-${OpenVINO_VERSION})
    endif()
    set(OV_CPACK_LICENSESDIR licenses)

    ov_get_pyversion(pyversion)
    if(pyversion)
        # should not be used in production; only by setup.py install
        set(OV_CPACK_PYTHONDIR lib/${pyversion}/site-packages)
    endif()

    # non-native stuff
    set(OV_CPACK_SHAREDIR ${CMAKE_INSTALL_DATADIR}/openvino) # internal
    set(OV_CPACK_SAMPLESDIR ${OV_CPACK_SHAREDIR}/samples)
    set(OV_CPACK_DEVREQDIR ${OV_CPACK_SHAREDIR})
    unset(OV_CPACK_SHAREDIR)

    # skipped during common libraries packaging
    set(OV_CPACK_WHEELSDIR "tools")
endmacro()

ov_common_libraries_cpack_set_dirs()

#
# Override CPack components name for common libraries generator
# This is needed to change the granularity, i.e. merge several components
# into a single one
#

macro(ov_override_component_names)
    # merge C++ and C runtimes
    set(OV_CPACK_COMP_CORE_C "${OV_CPACK_COMP_CORE}")
    set(OV_CPACK_COMP_CORE_C_DEV "${OV_CPACK_COMP_CORE_DEV}")
endmacro()

ov_override_component_names()

#
# Override include / exclude rules for components
# This is required to exclude some files from installation
# (e.g. debian packages don't require setupvars scripts)
#

macro(ov_define_component_include_rules)
    # core components
    unset(OV_CPACK_COMP_CORE_EXCLUDE_ALL)
    set(OV_CPACK_COMP_CORE_C_EXCLUDE_ALL ${OV_CPACK_COMP_CORE_EXCLUDE_ALL})
    unset(OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL)
    set(OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL ${OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL})
    # licensing
    if(CPACK_GENERATOR STREQUAL "CONAN")
        unset(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL)
    else()
        set(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    endif()
    # samples
    set(OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL ${OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL})
    set(OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # python
    set(OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_BENCHMARK_APP_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    set(OV_CPACK_COMP_OVC_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    # we don't pack artifacts of setup.py install, because it's called explicitly in conda / brew
    # or not used at all like in cases with conan / vcpkg
    set(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL ${OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL})
    # we don't need wheels in package, it's used installed only in open source distribution
    set(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # we don't need requirements.txt in package, because dependencies are installed by packages managers like conda
    set(OV_CPACK_COMP_OPENVINO_REQ_FILES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # tools
    set(OV_CPACK_COMP_OPENVINO_DEV_REQ_FILES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_DEPLOYMENT_MANAGER_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
endmacro()

ov_define_component_include_rules()

if(CPACK_GENERATOR STREQUAL "BREW")
    # brew relies on RPATH
    set(CMAKE_SKIP_INSTALL_RPATH OFF)
endif()
