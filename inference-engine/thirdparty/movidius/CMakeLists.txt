# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_MYRIAD)
    add_definitions(-DXLINK_USE_BUS)

    set(XLINK_DIR "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/XLink" CACHE PATH "path to Xlink")
    add_subdirectory("${XLINK_DIR}" "${CMAKE_BINARY_DIR}/thirdparty/movidius/XLink")

    add_subdirectory(
        "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc"
        "${CMAKE_BINARY_DIR}/thirdparty/movidius/mvnc")

endif()
