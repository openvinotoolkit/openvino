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
            ${CMAKE_SOURCE_DIR}/src/tests/functional/plugin/shared/include
            ${CMAKE_SOURCE_DIR}/src/tests/ie_test_utils/common_test_utils
        DEPENDENCIES
            openvino_intel_gpu_plugin
        LINK_LIBRARIES
            funcSharedTests
            OpenCL
        ADD_CPPLINT
        LABELS
            GPU
)

# try to find VA libraries
if(NOT ANDROID)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_search_module(libva QUIET IMPORTED_TARGET libva)
    endif()
endif()

if(libva_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::libva)
endif()

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
    target_link_libraries(${TARGET_NAME} PRIVATE d3d11 dxgi)
endif()
