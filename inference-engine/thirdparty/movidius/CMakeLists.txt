# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_MYRIAD)
    add_definitions(-DXLINK_USE_BUS)

    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9)
        set(XLINK_OLD_48_COMPILER ON)
    endif()
    # LTO usage leads to a linker fail on CentOS with the old 4.8.5 compiler during XLinkTests
    if(ENABLE_LTO AND NOT ENABLE_TESTS AND NOT XLINK_OLD_48_COMPILER)
	    ie_enable_lto()
    endif()

    set(XLINK_DIR "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/XLink" CACHE PATH "path to Xlink")
    add_subdirectory("${XLINK_DIR}" "${CMAKE_BINARY_DIR}/thirdparty/movidius/XLink")

    add_subdirectory(
        "${IE_MAIN_SOURCE_DIR}/thirdparty/movidius/mvnc"
        "${CMAKE_BINARY_DIR}/thirdparty/movidius/mvnc")

endif()
