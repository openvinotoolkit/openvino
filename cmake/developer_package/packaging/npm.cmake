# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

# We have to specify RPATH, all runtime libs are in one dir 
set(CMAKE_SKIP_INSTALL_RPATH OFF)

#
# ov_npm_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_npm_cpack_set_dirs)
    set(OV_CPACK_INCLUDEDIR .)
    set(OV_CPACK_OPENVINO_CMAKEDIR .)
    set(OV_CPACK_DOCDIR .)
    set(OV_CPACK_LICENSESDIR licenses)
    set(OV_CPACK_SAMPLESDIR .)
    set(OV_CPACK_WHEELSDIR .)
    set(OV_CPACK_TOOLSDIR .)
    set(OV_CPACK_DEVREQDIR .)
    set(OV_CPACK_PYTHONDIR .)

    set(OV_CPACK_LIBRARYDIR .)
    set(OV_CPACK_ARCHIVEDIR .)
    set(OV_CPACK_PLUGINSDIR .)

    set(OV_CPACK_RUNTIMEDIR .)
endmacro()

ov_npm_cpack_set_dirs()

#
# Override include / exclude rules for components
# This is required to exclude some files from installation
# (e.g. npm package requires only C++ Core component)
#

macro(ov_define_component_include_rules)
    # core components
    unset(OV_CPACK_COMP_CORE_EXCLUDE_ALL)
    set(OV_CPACK_COMP_CORE_C_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # tbb
    unset(OV_CPACK_COMP_TBB_EXCLUDE_ALL)
    set(OV_CPACK_COMP_TBB_DEV_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # licensing
    unset(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL)
    # samples
    set(OV_CPACK_COMP_CPP_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_C_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_PYTHON_SAMPLES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # python
    set(OV_CPACK_COMP_PYTHON_OPENVINO_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_BENCHMARK_APP_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_OVC_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_PYTHON_WHEELS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_OPENVINO_REQ_FILES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # nodejs
    unset(OV_CPACK_COMP_NPM_EXCLUDE_ALL)
    # tools
    set(OV_CPACK_COMP_OPENVINO_DEV_REQ_FILES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
endmacro()

ov_define_component_include_rules()

# New in version 3.18
set(CPACK_ARCHIVE_THREADS 8)
