# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 3.13)

project(OpenVINOConverter)

#
# Packages & settings
#

if(NOT DEFINED OpenVINO_SOURCE_DIR)
    get_filename_component(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." REALPATH)
endif()

if(NOT OpenVINODeveloperScripts_FOUND)
    find_package(OpenVINODeveloperScripts REQUIRED
                 PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
                 NO_CMAKE_FIND_ROOT_PATH
                 NO_DEFAULT_PATH)
endif()

#
# Installation rules
#

ov_get_pyversion(pyversion)
ov_cpack_add_component(${OV_CPACK_COMP_OVC}
                        HIDDEN)

install(DIRECTORY ${OpenVINOConverter_SOURCE_DIR}/openvino
        DESTINATION ${OV_CPACK_PYTHONDIR}
        COMPONENT ${OV_CPACK_COMP_OVC}
        ${OV_CPACK_COMP_OVC_EXCLUDE_ALL}
        USE_SOURCE_PERMISSIONS)

#
# Tests
#

if(ENABLE_TESTS)
    install(DIRECTORY unit_tests
            DESTINATION tests/ovc
            COMPONENT tests
            EXCLUDE_FROM_ALL)
endif()

#
# Cpack
#

if(CMAKE_SOURCE_DIR STREQUAL OpenVINOConverter_SOURCE_DIR)
    ov_cpack(${OV_CPACK_COMPONENTS_ALL})
endif()
