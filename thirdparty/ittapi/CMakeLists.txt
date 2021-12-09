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
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ittapi ${CMAKE_BINARY_DIR}/_deps/ittapi EXCLUDE_FROM_ALL)

        target_compile_definitions(ittnotify INTERFACE ENABLE_PROFILING_ITT)
        if(CMAKE_COMPILER_IS_GNUCXX)
            target_compile_options(ittnotify PRIVATE -Wno-undef)
        endif()

        # override INTERFACE_INCLUDE_DIRECTORIES
        set_property(TARGET ittnotify PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ittapi/src/ittnotify>
                     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ittapi/include>)

        openvino_developer_export_targets(COMPONENT openvino_common TARGETS ittnotify)
        ov_install_static_lib(ittnotify openvino_common)
    endif()
endif()
