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
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_DX11)
    target_link_libraries(${TARGET_NAME} PRIVATE d3d11 d3d12 dxgi dxguid)
endif()

find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    set(OV_GPU_FUNC_TESTS_VULKAN_TAG "v1.3.230" CACHE STRING "Git tag used when downloading Vulkan dependencies for GPU functional tests" FORCE)
    if(CMAKE_VERSION VERSION_LESS 3.14.0)
        message(WARNING "Vulkan was not found and automatic download requires CMake >= 3.14.0.")
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

        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(BUILD_WSI_XCB_SUPPORT OFF CACHE BOOL "" FORCE)
        set(BUILD_WSI_XLIB_SUPPORT OFF CACHE BOOL "" FORCE)
        set(BUILD_WSI_WAYLAND_SUPPORT OFF CACHE BOOL "" FORCE)
        set(UPDATE_DEPS OFF CACHE BOOL "" FORCE)
        
        # Vulkan-Loader's cJSON and asm_offset lack forward declarations,
        # which conflicts with OpenVINO's -Werror=missing-declarations.
        # Temporarily suppress this warning during FetchContent.
        set(_ov_vk_saved_c_flags "${CMAKE_C_FLAGS}")
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU"
           OR CMAKE_C_COMPILER_ID STREQUAL "Clang"
           OR CMAKE_C_COMPILER_ID STREQUAL "AppleClang"
           OR CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
            string(APPEND CMAKE_C_FLAGS " -Wno-missing-declarations -Wno-undef")
        elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
            # MSVC has no direct equivalent of -Wno-missing-declarations.
            # Lower warning level and silence #if-undef-identifier (C4668)
            # so the fetched Vulkan-Loader/cJSON sources do not break a /WX build.
            string(APPEND CMAKE_C_FLAGS " /W0 /wd4668")
        endif()
        
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
        set(CMAKE_C_FLAGS "${_ov_vk_saved_c_flags}")

        unset(BUILD_TESTS CACHE)
        unset(BUILD_WSI_XCB_SUPPORT CACHE)
        unset(BUILD_WSI_XLIB_SUPPORT CACHE)
        unset(BUILD_WSI_WAYLAND_SUPPORT CACHE)
        unset(UPDATE_DEPS CACHE)
        unset(VULKAN_HEADERS_ENABLE_TESTS CACHE)
        unset(VULKAN_HEADERS_ENABLE_INSTALL CACHE)

        if(TARGET vulkan)
            # Vulkan-Loader's vulkan-1.def hard-codes /OUT:vulkan-1.dll, but CMake
            # appends CMAKE_DEBUG_POSTFIX (e.g. 'd') in Debug, producing vulkan-1d.dll.
            # The mismatch raises LNK4070, which becomes a hard error under /WX.
            if(MSVC)
                target_link_options(vulkan PRIVATE /IGNORE:4070)
            endif()
            if(NOT TARGET Vulkan::Vulkan)
                add_library(Vulkan::Vulkan ALIAS vulkan)
            endif()
        endif()

        if(TARGET Vulkan::Vulkan)
            set(Vulkan_FOUND ON)
        endif()
    endif()
    if(UNIX AND NOT APPLE)
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
                        DESTINATION runtime/3rdparty/vulkan/lib
                        COMPONENT tests
                        EXCLUDE_FROM_ALL)
            endif()
        else()
            install(FILES "$<TARGET_FILE:Vulkan::Vulkan>"
                    DESTINATION runtime/3rdparty/vulkan/lib
                    COMPONENT tests
                    EXCLUDE_FROM_ALL)
        endif()
    endif()
endif()

if(Vulkan_FOUND)
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_VULKAN)
    target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Vulkan)
    if(TARGET Vulkan::Headers)
        target_link_libraries(${TARGET_NAME} PRIVATE Vulkan::Headers)
    endif()
else()
    message(FATAL_ERROR "Vulkan not found")
endif()

# Keep build-tree binaries relocatable so mounted paths (e.g. /ov in containers)
# still resolve local dependencies (libvulkan.so.1 etc.) from the executable directory.
if(UNIX AND NOT APPLE)
    set_property(TARGET ${TARGET_NAME} APPEND PROPERTY BUILD_RPATH "$ORIGIN")
endif()

ov_build_target_faster(${TARGET_NAME} PCH)
