# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gpuFuncTests)

addIeTargetTest(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            ov_intel_gpu_plugin
        LINK_LIBRARIES
            funcSharedTests
            OpenCL
        ADD_CPPLINT
        LABELS
            GPU
)

# try to find VA libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_search_module(LIBVA QUIET libva)
endif()

# TODO: pkg_search_module finds libva not in sysroot
if(ANDROID)
    set(LIBVA_FOUND OFF CACHE BOOL "" FORCE)
endif()

if(LIBVA_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_include_directories(${TARGET_NAME} PRIVATE ${LIBVA_INCLUDE_DIRS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${LIBVA_LINK_LIBRARIES})
endif()

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
    target_link_libraries(${TARGET_NAME} PRIVATE d3d11 dxgi)
endif()
