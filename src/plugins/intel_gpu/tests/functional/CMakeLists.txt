# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_gpu_func_tests)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4267, 4244 issues from oneDNN headers conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4267)
    ov_add_compiler_flags(/wd4244)
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
            openvino::reference
            funcSharedTests
            OpenCL::NewHeaders # should come before OpenCL::OpenCL
            OpenCL::OpenCL
        LABELS
            OV GPU
)

ov_gpu_set_runtime_interface_for(${TARGET_NAME})

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
endif()

if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
    target_link_libraries(${TARGET_NAME} PRIVATE d3d11 d3d12 dxgi dxguid)
endif()

find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    option(OV_GPU_FUNC_TESTS_FETCH_VULKAN "Download Vulkan-Headers and Vulkan-Loader for GPU functional tests when system Vulkan is unavailable" ON)
    if(OV_GPU_FUNC_TESTS_FETCH_VULKAN)
        set(OV_GPU_FUNC_TESTS_VULKAN_TAG "v1.4.350" CACHE STRING "Git tag used when downloading Vulkan dependencies for GPU functional tests" FORCE)
        if(CMAKE_VERSION VERSION_LESS 3.22.1)
            message(WARNING "Vulkan was not found and automatic download requires CMake >= 3.22.1. Vulkan-dependent GPU functional tests will remain unavailable.")
        else()
            include(FetchContent)

            # Use a short base directory and short content names to avoid hitting the
            # Windows MAX_PATH (260 chars) limit. FetchContent embeds the content name
            # multiple times into nested subbuild paths, so long names like
            # "ov_gpu_func_tests_vulkan_headers" easily overflow MAX_PATH on CI.
            set(_ov_vk_base_dir "${CMAKE_BINARY_DIR}/_vk")

            set(VULKAN_HEADERS_ENABLE_TESTS OFF)
            set(VULKAN_HEADERS_ENABLE_INSTALL OFF)
            FetchContent_Declare(
                ov_vk_headers
                GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
                GIT_TAG ${OV_GPU_FUNC_TESTS_VULKAN_TAG}
                GIT_SHALLOW TRUE
                SOURCE_DIR   "${_ov_vk_base_dir}/headers-src"
                BINARY_DIR   "${_ov_vk_base_dir}/headers-bld"
                SUBBUILD_DIR "${_ov_vk_base_dir}/headers-sub"
            )
            FetchContent_MakeAvailable(ov_vk_headers)
            string(REGEX REPLACE "^v" "" VulkanHeaders_VERSION "${OV_GPU_FUNC_TESTS_VULKAN_TAG}")
            if(NOT VulkanHeaders_VERSION MATCHES "^[0-9]+(\\.[0-9]+)*$")
                set(VulkanHeaders_VERSION "0.0.0")
            endif()

            set(BUILD_TESTS OFF)
            set(BUILD_WSI_XCB_SUPPORT OFF)
            set(BUILD_WSI_XLIB_SUPPORT OFF)
            set(BUILD_WSI_WAYLAND_SUPPORT OFF)
            set(UPDATE_DEPS OFF)
            FetchContent_Declare(
                ov_vk_loader
                GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
                GIT_TAG ${OV_GPU_FUNC_TESTS_VULKAN_TAG}
                GIT_SHALLOW TRUE
                SOURCE_DIR   "${_ov_vk_base_dir}/loader-src"
                BINARY_DIR   "${_ov_vk_base_dir}/loader-bld"
                SUBBUILD_DIR "${_ov_vk_base_dir}/loader-sub"
            )
            FetchContent_MakeAvailable(ov_vk_loader)

            unset(BUILD_TESTS)
            unset(BUILD_WSI_XCB_SUPPORT)
            unset(BUILD_WSI_XLIB_SUPPORT)
            unset(BUILD_WSI_WAYLAND_SUPPORT)
            unset(UPDATE_DEPS)
            unset(VULKAN_HEADERS_ENABLE_TESTS)
            unset(VULKAN_HEADERS_ENABLE_INSTALL)

            if(TARGET vulkan AND NOT TARGET Vulkan::Vulkan)
                add_library(Vulkan::Vulkan ALIAS vulkan)
            endif()

            if(TARGET Vulkan::Vulkan)
                set(Vulkan_FOUND ON)
            endif()
        endif()
    endif()
endif()

if(Vulkan_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_VULKAN)
    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
endif()

ov_build_target_faster(${TARGET_NAME} PCH)
