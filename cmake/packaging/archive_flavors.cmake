# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Per-run CPack project configuration for multi-flavor OpenVINO archives.
#
# This file is referenced by CPACK_PROJECT_CONFIG_FILE and is included by CPack
# once per package run, after CPackConfig.cmake. It lets a single build produce
# several distribution archives by redefining the component list per run.
#
# Flavor is selected via OV_CPACK_ARCHIVE_FLAVOR (default: full):
#   full   - complete OpenVINO archive (all components)
#   winget - reduced archive without samples and Python components
#
# Both flavors are produced from the same build by invoking cpack twice:
#   cpack -G TGZ -D OV_CPACK_ARCHIVE_FLAVOR=full
#   cpack -G TGZ -D OV_CPACK_ARCHIVE_FLAVOR=winget
#

if(NOT DEFINED OV_CPACK_ARCHIVE_FLAVOR)
    set(OV_CPACK_ARCHIVE_FLAVOR "full")
endif()

if(OV_CPACK_ARCHIVE_FLAVOR STREQUAL "full")
    set(CPACK_PACKAGE_FILE_NAME "openvino")
elseif(OV_CPACK_ARCHIVE_FLAVOR STREQUAL "winget")
    # explicit component list for the winget package (no samples, no Python);
    # components that are not built for the current configuration are ignored
    set(CPACK_COMPONENTS_ALL
        # core runtime and development files
        core
        core_c
        core_dev
        core_c_dev
        core_dev_links
        core_dev_pkgconfig
        # threading
        tbb
        tbb_dev
        # device plugins
        cpu
        gpu
        npu
        hetero
        multi
        batch
        # frontends
        ir
        onnx
        paddle
        pytorch
        tensorflow
        tensorflow_lite
        # licensing and scripts
        licensing
        setupvars
        install_dependencies)
    set(CPACK_PACKAGE_FILE_NAME "openvino_winget")
else()
    message(FATAL_ERROR "Unsupported OV_CPACK_ARCHIVE_FLAVOR '${OV_CPACK_ARCHIVE_FLAVOR}', expected 'full' or 'winget'")
endif()
