# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

# We have to specify RPATH, all runtime libs are in one dir
if(APPLE)
    # on macOS versions with SIP enabled, we need to use @rpath
    # because DYLD_LIBRARY_PATH is ignored
    set(CMAKE_SKIP_INSTALL_RPATH OFF)
else()
    # we don't need RPATHs, because setupvars.sh is used
    set(CMAKE_SKIP_INSTALL_RPATH ON)
endif()

#
# ov_wheel_cpack_set_dirs()
#
# Set directories for WHEEL cpack
#
macro(ov_wheel_cpack_set_dirs)
    set(OV_CPACK_INCLUDEDIR include)
    set(OV_CPACK_OPENVINO_CMAKEDIR cmake)
    set(OV_CPACK_DOCDIR docs)
    set(OV_CPACK_LICENSESDIR licenses)
    set(OV_CPACK_SAMPLESDIR samples)
    set(OV_CPACK_WHEELSDIR tools)
    set(OV_CPACK_TOOLSDIR tools)
    set(OV_CPACK_DEVREQDIR tools)
    set(OV_CPACK_PYTHONDIR python)
    set(OV_CPACK_LIBRARYDIR libs)
    set(OV_CPACK_RUNTIMEDIR libs)
    set(OV_CPACK_ARCHIVEDIR libs)

    set(OV_CPACK_PLUGINSDIR ${OV_CPACK_RUNTIMEDIR})
endmacro()

ov_wheel_cpack_set_dirs()

#
# Override include / exclude rules for components
# This is required to exclude some files from installation
# (e.g. archive packages don't require python_package component)
#

macro(ov_define_component_include_rules)
    # core components
    unset(OV_CPACK_COMP_CORE_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL)
    # c bindings
    unset(OV_CPACK_COMP_CORE_C_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL)
    # tbb
    unset(OV_CPACK_COMP_TBB_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_TBB_DEV_EXCLUDE_ALL)
    # licensing
    unset(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL)
    # samples
    set(OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # python
    unset(OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_BENCHMARK_APP_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_OVC_EXCLUDE_ALL)
    set(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    unset(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL)
    unset(OV_CPACK_COMP_OPENVINO_REQ_FILES_EXCLUDE_ALL)
    # nodejs
    set(OV_CPACK_COMP_NPM_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # tools
    set(OV_CPACK_COMP_OPENVINO_DEV_REQ_FILES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # links
    set(OV_CPACK_COMP_LINKS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
endmacro()

ov_define_component_include_rules()