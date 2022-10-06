# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (ENABLE_INTEL_MYRIAD)
    if(CMAKE_COMPILER_IS_GNUCC OR OV_COMPILER_IS_CLANG)
        ie_add_compiler_flags(-Wno-enum-conversion)
    endif()
    add_definitions(-DXLINK_USE_BUS)

    if(CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9)
        set(XLINK_OLD_48_COMPILER ON)
    endif()
    # LTO usage leads to a linker fail on CentOS with the old 4.8.5 compiler during XLinkTests
    if(ENABLE_LTO AND NOT ENABLE_TESTS AND NOT XLINK_OLD_48_COMPILER)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})
    endif()

    set(XLINK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/XLink" CACHE PATH "path to Xlink" FORCE)
    add_subdirectory(
        "${XLINK_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/XLink")

    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/mvnc"
        "${CMAKE_CURRENT_BINARY_DIR}/mvnc")

    if(ENABLE_TESTS)
        add_subdirectory(tests)
    endif()
endif()
