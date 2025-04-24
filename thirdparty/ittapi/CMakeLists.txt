# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(DEFINED INTEL_VTUNE_DIR OR DEFINED ENV{INTEL_VTUNE_DIR})
    find_package(ITT
                 PATHS "${OpenVINO_SOURCE_DIR}/src/common/itt/cmake"
                 NO_DEFAULT_PATH)
    if(NOT ITT_FOUND)
        message(WARNING "Profiling option enabled, but no ITT library was found under INTEL_VTUNE_DIR")
    endif()
else()
    add_subdirectory(ittapi EXCLUDE_FROM_ALL)

    target_compile_definitions(ittnotify INTERFACE ENABLE_PROFILING_ITT)
    if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
        target_compile_options(ittnotify PRIVATE -Wno-undef)
    endif()

    # override INTERFACE_INCLUDE_DIRECTORIES
    set_property(TARGET ittnotify PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ittapi/src/ittnotify>
                    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ittapi/include>)

    # create alias ittapi::ittnotify
    add_library(ittapi::ittnotify ALIAS ittnotify)

    ov_developer_package_export_targets(TARGET ittapi::ittnotify)
    ov_install_static_lib(ittnotify ${OV_CPACK_COMP_CORE})
endif()
