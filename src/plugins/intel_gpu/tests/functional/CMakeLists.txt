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

if(NOT WIN32)
    find_package(Vulkan QUIET)
endif()
if(NOT Vulkan_FOUND AND NOT WIN32)
    set(OV_GPU_FUNC_TESTS_VULKAN_TAG "v1.3.230" CACHE STRING "Git tag used when downloading Vulkan dependencies for GPU functional tests" FORCE)
    if(CMAKE_VERSION VERSION_LESS 3.14.0)
        message(WARNING "Vulkan was not found and automatic download requires CMake >= 3.14.0.")
    else()
        include(FetchContent)

        set(VULKAN_HEADERS_ENABLE_TESTS OFF)
        set(VULKAN_HEADERS_ENABLE_INSTALL OFF)
        FetchContent_Declare(
            ov_vk_headers
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
            GIT_TAG ${OV_GPU_FUNC_TESTS_VULKAN_TAG}
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(ov_vk_headers)
        string(REGEX REPLACE "^v" "" VulkanHeaders_VERSION "${OV_GPU_FUNC_TESTS_VULKAN_TAG}")
        if(NOT VulkanHeaders_VERSION MATCHES "^[0-9]+(\\.[0-9]+)*$")
            set(VulkanHeaders_VERSION "0.0.0")
        endif()

        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_WSI_XCB_SUPPORT OFF CACHE BOOL "" FORCE)
        set(BUILD_WSI_XLIB_SUPPORT OFF CACHE BOOL "" FORCE)
        set(BUILD_WSI_WAYLAND_SUPPORT OFF CACHE BOOL "" FORCE)
        set(UPDATE_DEPS OFF CACHE BOOL "" FORCE)
        
        FetchContent_Declare(
            ov_vk_loader
            GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Loader.git
            GIT_TAG ${OV_GPU_FUNC_TESTS_VULKAN_TAG}
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(ov_vk_loader)
        foreach(_ov_vk_tgt vulkan asm_offset)
            if(CMAKE_C_COMPILER_ID STREQUAL "GNU"
               OR CMAKE_C_COMPILER_ID STREQUAL "Clang"
               OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang"
               OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
                # Vulkan-Loader's cJSON and asm_offset trigger warnings that are
                # promoted to errors in OpenVINO builds;
                target_compile_options(${_ov_vk_tgt} PRIVATE
                                       -Wno-missing-declarations
                                       -Wno-undef
                                       -Wno-typedef-redefinition)
            endif()
        endforeach()

        unset(BUILD_TESTS CACHE)
        unset(BUILD_WSI_XCB_SUPPORT CACHE)
        unset(BUILD_WSI_XLIB_SUPPORT CACHE)
        unset(BUILD_WSI_WAYLAND_SUPPORT CACHE)
        unset(UPDATE_DEPS CACHE)
        unset(VULKAN_HEADERS_ENABLE_TESTS CACHE)
        unset(VULKAN_HEADERS_ENABLE_INSTALL CACHE)

        if(TARGET vulkan)
            if(NOT TARGET Vulkan::Vulkan)
                add_library(Vulkan::Vulkan ALIAS vulkan)
            endif()
        endif()

        if(TARGET Vulkan::Vulkan)
            set(Vulkan_FOUND ON)
        endif()
    endif()
    if(UNIX AND NOT APPLE)
        set(_ov_vk_install_dir runtime/3rdparty/vulkan/lib)
        # Install Vulkan loader next to other bundled 3rdparty runtimes so
        # setupvars can expose it for install-tree test execution.
        get_target_property(_ov_vk_imported Vulkan::Vulkan IMPORTED)
        if(_ov_vk_imported)
            get_target_property(_ov_vk_lib_location Vulkan::Vulkan IMPORTED_LOCATION)
            if(NOT _ov_vk_lib_location)
                get_target_property(_ov_vk_lib_location Vulkan::Vulkan IMPORTED_LOCATION_RELEASE)
            endif()
            if(NOT _ov_vk_lib_location)
                get_target_property(_ov_vk_lib_location Vulkan::Vulkan IMPORTED_LOCATION_RELWITHDEBINFO)
            endif()
            if(NOT _ov_vk_lib_location)
                get_target_property(_ov_vk_lib_location Vulkan::Vulkan IMPORTED_LOCATION_DEBUG)
            endif()

            if(_ov_vk_lib_location)
                install(FILES "${_ov_vk_lib_location}"
                        DESTINATION ${_ov_vk_install_dir}
                        COMPONENT tests)
                install(FILES "${_ov_vk_lib_location}"
                        DESTINATION ${_ov_vk_install_dir}
                        RENAME libvulkan.so.1
                        COMPONENT tests)
                install(FILES "${_ov_vk_lib_location}"
                        DESTINATION ${_ov_vk_install_dir}
                        RENAME libvulkan.so
                        COMPONENT tests)
            endif()
        else()
            install(FILES "$<TARGET_FILE:Vulkan::Vulkan>"
                    DESTINATION ${_ov_vk_install_dir}
                    COMPONENT tests)
            install(FILES "$<TARGET_FILE:Vulkan::Vulkan>"
                    DESTINATION ${_ov_vk_install_dir}
                    RENAME libvulkan.so.1
                    COMPONENT tests)
            install(FILES "$<TARGET_FILE:Vulkan::Vulkan>"
                    DESTINATION ${_ov_vk_install_dir}
                    RENAME libvulkan.so
                    COMPONENT tests)
        endif()
    endif()
endif()

if(Vulkan_FOUND AND NOT WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_VULKAN)
    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
    if(TARGET Vulkan::Headers)
        target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Headers)
    endif()
elseif(NOT WIN32)
    message(FATAL_ERROR "Vulkan not found")
endif()

# Keep build-tree binaries relocatable so mounted paths (e.g. /ov in containers)
# still resolve local dependencies (libvulkan.so.1 etc.) from the executable directory.
if(UNIX AND NOT APPLE)
    set_property(TARGET ${TARGET_NAME} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
endif()

ov_build_target_faster(${TARGET_NAME} PCH)
