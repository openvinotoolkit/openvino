# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(GNUInstallDirs)

set(CMAKE_SKIP_INSTALL_RPATH OFF)

#
# ov_npm_cpack_set_dirs()
#
# Set directories for cpack
#
macro(ov_npm_cpack_set_dirs)
    set(OV_CPACK_INCLUDEDIR runtime/include)
    set(OV_CPACK_IE_CMAKEDIR runtime/cmake)
    set(OV_CPACK_NGRAPH_CMAKEDIR runtime/cmake)
    set(OV_CPACK_OPENVINO_CMAKEDIR runtime/cmake)
    set(OV_CPACK_DOCDIR docs)
    set(OV_CPACK_LICENSESDIR licenses)
    set(OV_CPACK_SAMPLESDIR samples)
    set(OV_CPACK_WHEELSDIR tools)
    set(OV_CPACK_TOOLSDIR tools)
    set(OV_CPACK_DEVREQDIR tools)
    set(OV_CPACK_PYTHONDIR python)

    if(WIN32)
        set(OV_CPACK_LIBRARYDIR ${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_RUNTIMEDIR ${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_ARCHIVEDIR ${ARCH_FOLDER}/${build_type})
    elseif(APPLE)
        set(OV_CPACK_LIBRARYDIR ${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_RUNTIMEDIR ${ARCH_FOLDER}/${build_type})
        set(OV_CPACK_ARCHIVEDIR ${ARCH_FOLDER}/${build_type})
    else()
        set(OV_CPACK_LIBRARYDIR ${ARCH_FOLDER})
        set(OV_CPACK_RUNTIMEDIR ${ARCH_FOLDER})
        set(OV_CPACK_ARCHIVEDIR ${ARCH_FOLDER})
    endif()
endmacro()

ov_npm_cpack_set_dirs()

#
# Override include / exclude rules for components
# This is required to exclude some files from installation
# (e.g. npm package requires only C++ Core component)
#

macro(ov_define_component_include_rules)
    # core components
    # unset(OV_CPACK_COMP_CORE_EXCLUDE_ALL)
    set(OV_CPACK_COMP_CORE_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_CORE_C_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_CORE_DEV_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_CORE_C_DEV_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # licensing
    set(OV_CPACK_COMP_LICENSING_EXCLUDE_ALL EXCLUDE_FROM_ALL)
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
    set(OV_CPACK_COMP_DEPLOYMENT_MANAGER_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    # scripts
    set(OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL EXCLUDE_FROM_ALL)
    set(OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL EXCLUDE_FROM_ALL)
endmacro()

ov_define_component_include_rules()

# New in version 3.18
set(CPACK_ARCHIVE_THREADS 8)
