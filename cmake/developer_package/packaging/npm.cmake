# Copyright (C) 2018-2024 Intel Corporation
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
    set(OV_CPACK_DEVREQDIR .)
    set(OV_CPACK_PYTHONDIR .)

    set(OV_CPACK_LIBRARYDIR .)
    set(OV_CPACK_ARCHIVEDIR .)
    set(OV_CPACK_PLUGINSDIR .)

    set(OV_CPACK_RUNTIMEDIR .)
endmacro()

ov_npm_cpack_set_dirs()

#
# Override CPack components name for NPM generator
# This is needed to change the granularity, i.e. merge several components
# into a single one
#

macro(ov_override_component_names)
    # merge links and pkgconfig with dev component
    set(OV_CPACK_COMP_LINKS "${OV_CPACK_COMP_CORE_DEV}")
    set(OV_CPACK_COMP_PKG_CONFIG "${OV_CPACK_COMP_CORE_DEV}")
endmacro()

ov_override_component_names()

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
    # openmp
    unset(OV_CPACK_COMP_OPENMP_EXCLUDE_ALL)
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
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # pkgconfig
    unset(OV_CPACK_COMP_PKG_CONFIG_EXCLUDE_ALL)
    # symbolic links
    unset(OV_CPACK_COMP_LINKS_EXCLUDE_ALL)
    # npu internal tools
    set(OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL EXCLUDE_FROM_ALL)
endmacro()

ov_define_component_include_rules()

# New in version 3.18
set(CPACK_ARCHIVE_THREADS 8)
