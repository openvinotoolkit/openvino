# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

foreach(var PKG_CONFIG_IN_FILE PKG_CONFIG_OUT_FILE
            PKGCONFIG_OpenVINO_PREFIX OV_CPACK_RUNTIMEDIR
            OV_CPACK_INCLUDEDIR OpenVINO_VERSION
            PKGCONFIG_OpenVINO_DEFINITIONS
            PKGCONFIG_OpenVINO_FRONTENDS
            PKGCONFIG_OpenVINO_PRIVATE_DEPS)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "Variable ${var} is not defined")
    endif()
endforeach()

# create command

if(NOT EXISTS "${PKG_CONFIG_IN_FILE}")
    message(FATAL_ERROR "${PKG_CONFIG_IN_FILE} does not exist")
endif()

# execute

configure_file("${PKG_CONFIG_IN_FILE}" "${PKG_CONFIG_OUT_FILE}" @ONLY)
