# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_PROFILING_ITT)
    if(DEFINED INTEL_VTUNE_DIR OR DEFINED ENV{INTEL_VTUNE_DIR})
        find_package(ITT
                     PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
                     NO_DEFAULT_PATH)
        if(NOT ITT_FOUND)
            message(WARNING "Profiling option enabled, but no ITT library was found under INTEL_VTUNE_DIR")
        endif()
    else()
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ittapi ${CMAKE_BINARY_DIR}/_deps/ittapi)

        target_compile_definitions(ittnotify INTERFACE ENABLE_PROFILING_ITT)
        if (UNIX)
            target_compile_options(ittnotify PRIVATE -Wno-undef)
        endif()

        openvino_developer_export_targets(COMPONENT openvino_common TARGETS ittnotify)
    endif()
endif()
