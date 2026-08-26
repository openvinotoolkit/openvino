# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gpu_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267 issues from oneDNN headers conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4267)
    # 'initializing': truncation from 'XXX' to 'YYY'
    ov_add_compiler_flags(/wd4305)
endif()

list(APPEND DEFINES TEST_CUSTOM_OP_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/custom_op/custom_op.xml")

ov_add_test_target(
        NAME
            ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${TEST_COMMON_SOURCE_DIR}
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            $<TARGET_PROPERTY:openvino_intel_gpu_plugin,SOURCE_DIR>/include/
            ${TEST_COMMON_INCLUDE_DIR}
        DEFINES
            ${DEFINES}
        DEPENDENCIES
            openvino_intel_gpu_plugin
        LINK_LIBRARIES
            funcSharedTests
            openvino::reference
            OpenCL::NewHeaders # should come before OpenCL::OpenCL
            OpenCL::OpenCL
        LABELS
            OV GPU
)

# Functional tests load the plugin dynamically; OV_GPU_WITH_*_RT only selects which remote-tensor
# interop tests compile. COMBINED ships both runtimes, so compile both sets; single-RT its own.
if(GPU_RT_TYPE STREQUAL "COMBINED")
    target_compile_definitions(${TARGET_NAME} PRIVATE OV_GPU_WITH_ZE_RT=1 OV_GPU_WITH_OCL_RT=1)
    add_dependencies(${TARGET_NAME} openvino_intel_gpu_plugin_ocl)
else()
    ov_gpu_set_runtime_interface_for(${TARGET_NAME} RUNTIME ${GPU_RT_TYPE})
endif()

if(ENABLE_PROXY)
    target_compile_definitions(${TARGET_NAME} PUBLIC PROXY_PLUGIN_ENABLED)
endif()

# try to find VA libraries
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_search_module(libva QUIET IMPORTED_TARGET libva)
endif()

if(libva_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA)
    target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::libva)

    # VA surface interop tests need a VADisplay of their own. The DRM backend is the only one
    # which works headless, so these tests are enabled only when libva-drm is available.
    pkg_search_module(libva_drm QUIET IMPORTED_TARGET libva-drm)
    if(libva_drm_FOUND AND EXISTS "${libva_INCLUDEDIR}/va/va.h")
        target_include_directories(${TARGET_NAME} PRIVATE "${libva_INCLUDEDIR}/va")
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_LIBVA_DRM OV_GPU_USE_SYSTEM_LIBVA_HEADERS)
        target_link_libraries(${TARGET_NAME} PRIVATE PkgConfig::libva_drm)
    endif()
endif()

if(WIN32)
    include(CheckIncludeFileCXX)
    # DX11 and DX12 SDK headers may be available independently; enable each
    # interop test set only when its corresponding header is present to avoid
    # build breaks on environments that ship only one of the SDKs.
    check_include_file_cxx(d3d11.h OV_GPU_FUNC_TESTS_HAVE_D3D11_H)
    check_include_file_cxx(d3d12.h OV_GPU_FUNC_TESTS_HAVE_D3D12_H)

    if(OV_GPU_FUNC_TESTS_HAVE_D3D11_H)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
        target_link_libraries(${TARGET_NAME} PRIVATE d3d11)
    endif()
    if(OV_GPU_FUNC_TESTS_HAVE_D3D12_H)
        target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX12)
        target_link_libraries(${TARGET_NAME} PRIVATE d3d12)
    endif()
    if(OV_GPU_FUNC_TESTS_HAVE_D3D11_H OR OV_GPU_FUNC_TESTS_HAVE_D3D12_H)
        target_link_libraries(${TARGET_NAME} PRIVATE dxgi dxguid)
    endif()
endif()

ov_build_target_faster(${TARGET_NAME} PCH)
