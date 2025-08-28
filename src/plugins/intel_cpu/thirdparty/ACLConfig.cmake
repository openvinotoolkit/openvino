# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ExternalProject)

# ==============================================================================
# ARM Compute Library Configuration
# ==============================================================================
# This file configures ARM Compute Library (ACL) for OpenVINO CPU plugin.
# It supports three modes:
# 1. Using pre-built ACL library (ARM_COMPUTE_LIB_DIR is set)
# 2. Building ACL using CMake (ENABLE_ARM_COMPUTE_CMAKE is set)
# 3. Building ACL using SCons (default)
# ==============================================================================

# ------------------------------------------------------------------------------
# Mode 1: Use pre-built ARM Compute Library
# ------------------------------------------------------------------------------
if(ARM_COMPUTE_INCLUDE_DIR OR ARM_COMPUTE_LIB_DIR)
    set(ARM_COMPUTE_INCLUDE_DIR "" CACHE PATH "Path to ARM Compute Library headers" FORCE)

    if(NOT ARM_COMPUTE_LIB_DIR)
        message(FATAL_ERROR "Undefined ARM_COMPUTE_LIB_DIR input variable should be set manually")
    endif()

    if(NOT TARGET arm_compute::arm_compute)
        # Find library with proper configuration
        if(WIN32 OR APPLE)
            if(OV_GENERATOR_MULTI_CONFIG)
                set(extra_args PATH_SUFFIXES ${CMAKE_CONFIGURATION_TYPES})
            else()
                set(extra_args PATH_SUFFIXES ${CMAKE_BUILD_TYPE})
            endif()
        endif()

        find_library(ARM_COMPUTE_LIB
                     NAMES arm_compute-static
                     PATHS ${ARM_COMPUTE_LIB_DIR}
                     ${extra_args})
        unset(extra_args)

        message(STATUS "Found arm_compute-static: ${ARM_COMPUTE_LIB}")

        # Create imported targets
        add_library(arm_compute::arm_compute STATIC IMPORTED GLOBAL)
        set_target_properties(arm_compute::arm_compute PROPERTIES
            IMPORTED_LOCATION ${ARM_COMPUTE_LIB})

        add_library(arm_compute::half INTERFACE IMPORTED GLOBAL)

        if(ARM_COMPUTE_INCLUDE_DIR)
            set_target_properties(arm_compute::arm_compute arm_compute::half PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_INCLUDE_DIR})
        endif()
    endif()

    # Setup for oneDNN integration
    set(ACL_FOUND ON)
    set(ACL_LIBRARIES arm_compute::arm_compute arm_compute::half)

    foreach(acl_library IN LISTS ACL_LIBRARIES)
        list(APPEND ACL_INCLUDE_DIRS
                $<TARGET_PROPERTY:${acl_library},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()

# ------------------------------------------------------------------------------
# Mode 2: Build ACL using CMake
# ------------------------------------------------------------------------------
elseif(ENABLE_ARM_COMPUTE_CMAKE)
    set(ARM_COMPUTE_SOURCE_DIR "${intel_cpu_thirdparty_SOURCE_DIR}/ComputeLibrary")
    set(ARM_COMPUTE_BINARY_DIR "${intel_cpu_thirdparty_BINARY_DIR}/ComputeLibrary")

    function(ov_build_compute_library)
        # Configure ComputeLibrary build
        set(BUILD_SHARED_LIBS OFF)
        set(ARM_COMPUTE_GRAPH_ENABLED OFF CACHE BOOL "" FORCE)
        set(OPENMP OFF CACHE BOOL "" FORCE)
        set(CPPTHREADS OFF CACHE BOOL "" FORCE)

        # SVE is not supported on Darwin
        if(CMAKE_HOST_APPLE)
            set(ENABLE_SVE OFF CACHE BOOL "" FORCE)
            set(ARM_COMPUTE_ENABLE_SVE OFF CACHE BOOL "" FORCE)
            set(ARM_COMPUTE_ENABLE_SVEF32MM OFF CACHE BOOL "" FORCE)
        endif()

        add_subdirectory(${ARM_COMPUTE_SOURCE_DIR} ${ARM_COMPUTE_BINARY_DIR} EXCLUDE_FROM_ALL)

        add_library(ArmCompute::Half INTERFACE IMPORTED GLOBAL)
        set_target_properties(ArmCompute::Half PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ARM_COMPUTE_SOURCE_DIR}/include")
    endfunction()

    ov_build_compute_library()

    # Setup for oneDNN integration
    set(ACL_FOUND ON)
    set(ACL_LIBRARIES arm_compute_core ArmCompute::Half)

    foreach(acl_library IN LISTS ACL_LIBRARIES)
        list(APPEND ACL_INCLUDE_DIRS
                $<TARGET_PROPERTY:${acl_library},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()

    set(ENV{ACL_ROOT_DIR} "${ARM_COMPUTE_SOURCE_DIR}")

# ------------------------------------------------------------------------------
# Mode 3: Build ACL using SCons (default)
# ------------------------------------------------------------------------------
elseif(NOT TARGET arm_compute::arm_compute)

    # ==========================================================================
    # Initial Setup
    # ==========================================================================

    set(ARM_COMPUTE_SCONS_JOBS "8" CACHE STRING "Number of parallel threads to build ARM Compute Library")
    set(ARM_COMPUTE_SOURCE_DIR "${intel_cpu_thirdparty_SOURCE_DIR}/ComputeLibrary")

    message(STATUS "Configure to build ${ARM_COMPUTE_SOURCE_DIR}")

    # Find SCons build tool
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        list(APPEND find_scons_extra_options REQUIRED)
    endif()

    find_host_program(SCONS scons ${find_scons_extra_options})
    if(NOT SCONS)
        message(FATAL_ERROR "Scons tool is not found!")
    endif()

    # Set build directory
    if(DEFINED intel_cpu_thirdparty_BINARY_DIR)
        set(ARM_COMPUTE_BUILD_DIR "${intel_cpu_thirdparty_BINARY_DIR}/acl_build" CACHE PATH "Path to ARM Compute Library build directory")
    else()
        set(ARM_COMPUTE_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/acl_build" CACHE PATH "Path to ARM Compute Library build directory")
    endif()

    # ==========================================================================
    # Helper Functions
    # ==========================================================================

    # Helper function to add ARM Compute options with consistent formatting
    function(ov_arm_compute_add_option key value)
        list(APPEND ARM_COMPUTE_OPTIONS "${key}=${value}")
        set(ARM_COMPUTE_OPTIONS "${ARM_COMPUTE_OPTIONS}" PARENT_SCOPE)
    endfunction()

    # Helper function to configure platform-specific settings
    function(ov_arm_compute_configure_platform)
        # Android configuration
        if(ANDROID)
            if(ANDROID_PLATFORM_LEVEL LESS 18)
                message(FATAL_ERROR "ARM compute library requires Android API 18 level and higher"
                                    "Please, specify -DANDROID_PLATFORM=android-18 at least")
            endif()

            if(ANDROID_NDK_REVISION LESS "23.0")
                ov_arm_compute_add_option("toolchain_prefix" "${ANDROID_TOOLCHAIN_PREFIX}")
            else()
                string(REGEX REPLACE "/bin/[^/]+-" "/bin/llvm-" ANDROID_TOOLCHAIN_PREFIX_FIXED "${ANDROID_TOOLCHAIN_PREFIX}")
                message(STATUS "SCONS: using ANDROID_TOOLCHAIN_PREFIX=${ANDROID_TOOLCHAIN_PREFIX_FIXED}")
                ov_arm_compute_add_option("toolchain_prefix" "${ANDROID_TOOLCHAIN_PREFIX_FIXED}")
            endif()

            ov_arm_compute_add_option("compiler_prefix" "${ANDROID_TOOLCHAIN_ROOT}/bin/")

            set(extra_cc_flags "--target=${ANDROID_LLVM_TRIPLE}" PARENT_SCOPE)
            set(extra_flags "${extra_flags} --gcc-toolchain=${ANDROID_TOOLCHAIN_ROOT}" PARENT_SCOPE)
            set(extra_flags "${extra_flags} --sysroot=${CMAKE_SYSROOT}" PARENT_SCOPE)

        # Linux configuration
        elseif(LINUX)
            get_filename_component(cxx_compiler "${CMAKE_CXX_COMPILER}" NAME)
            get_filename_component(c_compiler "${CMAKE_C_COMPILER}" NAME)
            get_filename_component(compiler_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)

            set(cmake_build_env
                CC=${c_compiler}
                CXX=${cxx_compiler} PARENT_SCOPE)
            ov_arm_compute_add_option("compiler_prefix" "${compiler_prefix}/")

        # Emscripten configuration
        elseif(EMSCRIPTEN)
            set(cmake_build_env
                CC=emcc
                CXX=em++
                RANLIB=emranlib
                AR=emar PARENT_SCOPE)

            get_filename_component(toolchain_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)
            ov_arm_compute_add_option("toolchain_prefix" "${toolchain_prefix}/")

        # Apple/macOS configuration
        elseif(APPLE)
            get_filename_component(cxx_compiler "${CMAKE_CXX_COMPILER}" NAME)
            get_filename_component(c_compiler "${CMAKE_C_COMPILER}" NAME)
            get_filename_component(compiler_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)

            set(cmake_build_env
                CC=${c_compiler}
                CXX=${cxx_compiler} PARENT_SCOPE)

            ov_arm_compute_add_option("compiler_prefix" "${compiler_prefix}/")
        endif()

        set(ARM_COMPUTE_OPTIONS "${ARM_COMPUTE_OPTIONS}" PARENT_SCOPE)
    endfunction()

    # Helper function to configure compiler flags
    function(ov_arm_compute_configure_flags)
        set(local_extra_cxx_flags "${extra_cxx_flags}")
        set(local_extra_link_flags "${extra_link_flags}")
        set(local_extra_cc_flags "${extra_cc_flags}")

        # Platform-specific flags
        if(ANDROID)
            set(extra_flags "${extra_flags} --gcc-toolchain=${ANDROID_TOOLCHAIN_ROOT}")
            set(extra_flags "${extra_flags} --sysroot=${CMAKE_SYSROOT}")
            set(local_extra_link_flags "${local_extra_link_flags} ${extra_flags}")
            set(local_extra_cxx_flags "${local_extra_cxx_flags} ${extra_flags}")

        elseif(EMSCRIPTEN)
            # EMSDK: Passing any of -msse, -msse2, -msse3, -mssse3, -msse4.1, -msse4.2,
            # -msse4, -mavx, -mfpu=neon flags also requires passing -msimd128 (or -mrelaxed-simd)!
            set(local_extra_cxx_flags "${local_extra_cxx_flags} -msimd128")
            # clang-16: error: argument unused during compilation: '-mthumb' [-Werror,-Wunused-command-line-argument]
            # clang-16: error: argument unused during compilation: '-mfloat-abi=hard' [-Werror,-Wunused-command-line-argument]
            set(local_extra_cxx_flags "${local_extra_cxx_flags} \
                -Wno-unused-command-line-argument \
                -Wno-unknown-warning-option \
                -Wno-unused-function \
                -Wno-unused-but-set-variable")

        elseif(APPLE)
            if(CMAKE_OSX_DEPLOYMENT_TARGET)
                set(local_extra_cxx_flags "${local_extra_cxx_flags} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
            endif()

            if(HOST_X86_64 AND CMAKE_OSX_SYSROOT)
                set(local_extra_cxx_flags "${local_extra_cxx_flags} --sysroot ${CMAKE_OSX_SYSROOT}")
            endif()

            if(OV_COMPILER_IS_CLANG)
                set(local_extra_cxx_flags "${local_extra_cxx_flags} -Wno-error=return-stack-address")
            endif()

            if(CMAKE_OSX_ARCHITECTURES)
                foreach(arch IN LISTS CMAKE_OSX_ARCHITECTURES)
                    set(local_extra_cxx_flags "${local_extra_cxx_flags} -arch ${arch}")
                endforeach()
            endif()

        elseif(MSVC64)
            if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
                set(local_extra_cxx_flags "${local_extra_cxx_flags} $<IF:$<CONFIG:Release>,/MD,/MDd>")
            else()
                if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                    set(local_extra_cxx_flags "${local_extra_cxx_flags} /MDd")
                else()
                    set(local_extra_cxx_flags "${local_extra_cxx_flags} /MD")
                endif()
            endif()
        endif()

        # LTO configuration
        if(ENABLE_LTO AND NOT CMAKE_CROSSCOMPILING)
            if(CMAKE_COMPILER_IS_GNUCXX)
                set(local_extra_cxx_flags "${local_extra_cxx_flags} -flto -fno-fat-lto-objects")
                set(local_extra_link_flags "${local_extra_link_flags} -flto -fno-fat-lto-objects")
            elseif(OV_COMPILER_IS_CLANG)
                set(local_extra_cxx_flags "${local_extra_cxx_flags} -flto=thin")
                set(local_extra_link_flags "${local_extra_link_flags} -flto=thin")
            endif()
        endif()

        # Warning suppression
        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(local_extra_cxx_flags "${local_extra_cxx_flags} -Wno-suggest-override")
        endif()

        # Multi-ISA support
        if(NOT ARM AND OV_CPU_AARCH64_USE_MULTI_ISA)
            set(local_extra_cxx_flags "${local_extra_cxx_flags} -DENABLE_SME -DARM_COMPUTE_ENABLE_SME -DARM_COMPUTE_ENABLE_SME2")
        endif()

        # Export flags
        set(extra_cxx_flags "${local_extra_cxx_flags}" PARENT_SCOPE)
        set(extra_link_flags "${local_extra_link_flags}" PARENT_SCOPE)
        set(extra_cc_flags "${local_extra_cc_flags}" PARENT_SCOPE)
    endfunction()

    # ==========================================================================
    # Configure Build Options
    # ==========================================================================

    set(ARM_COMPUTE_OPTIONS)

    # Basic configuration options
    ov_arm_compute_add_option("neon" "1")
    ov_arm_compute_add_option("opencl" "0")
    ov_arm_compute_add_option("examples" "0")
    ov_arm_compute_add_option("Werror" "0")
    ov_arm_compute_add_option("gemm_tuner" "0")
    ov_arm_compute_add_option("reference_openmp" "0")
    ov_arm_compute_add_option("validation_tests" "0")
    ov_arm_compute_add_option("benchmark_tests" "0")
    ov_arm_compute_add_option("data_layout_support" "all")
    ov_arm_compute_add_option("arch" "${OV_CPU_ARM_TARGET_ARCH}")
    ov_arm_compute_add_option("build_dir" "${OV_CPU_ARM_TARGET_ARCH}")

    # Threading configuration
    if(THREADING STREQUAL "OMP")
        ov_arm_compute_add_option("openmp" "1")
        ov_arm_compute_add_option("cppthreads" "0")
    else()
        ov_arm_compute_add_option("openmp" "0")
        ov_arm_compute_add_option("cppthreads" "1")
    endif()

    # Architecture configuration
    if(ARM)
        ov_arm_compute_add_option("estate" "32")
    else()
        ov_arm_compute_add_option("estate" "64")
        if(OV_CPU_AARCH64_USE_MULTI_ISA)
            ov_arm_compute_add_option("multi_isa" "1")
        endif()
    endif()

    # Install directory
    if(NOT MSVC64)
        ov_arm_compute_add_option("install_dir" "install")
    endif()

    # Build jobs
    if(ARM_COMPUTE_SCONS_JOBS)
        ov_arm_compute_add_option("--jobs" "${ARM_COMPUTE_SCONS_JOBS}")
    endif()

    # Debug options
    set(ARM_COMPUTE_DEBUG_OPTIONS "debug=1" "asserts=1" "logging=1")

    if(OV_GENERATOR_MULTI_CONFIG AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
        foreach(option IN LISTS ARM_COMPUTE_DEBUG_OPTIONS)
            list(APPEND ARM_COMPUTE_OPTIONS
                $<$<CONFIG:Debug>:${option}>
                $<$<CONFIG:RelWithDebInfo>:${option}>)
        endforeach()
    elseif(CMAKE_BUILD_TYPE MATCHES "^(Debug|RelWithDebInfo)$")
        foreach(option IN LISTS ARM_COMPUTE_DEBUG_OPTIONS)
            list(APPEND ARM_COMPUTE_OPTIONS ${option})
        endforeach()
    endif()

    # OS configuration
    if(EMSCRIPTEN OR LINUX)
        ov_arm_compute_add_option("os" "linux")
    elseif(ANDROID)
        ov_arm_compute_add_option("os" "android")
    elseif(APPLE)
        ov_arm_compute_add_option("os" "macos")
    elseif(WIN32)
        ov_arm_compute_add_option("os" "windows")
    endif()

    # Build type
    if(CMAKE_CROSSCOMPILING)
        ov_arm_compute_add_option("build" "cross_compile")
    else()
        ov_arm_compute_add_option("build" "native")
    endif()

    # Compiler cache
    if(CMAKE_CXX_COMPILER_LAUNCHER)
        ov_arm_compute_add_option("compiler_cache" "${CMAKE_CXX_COMPILER_LAUNCHER}")
    endif()

    # Toolchain prefix (for Yocto builds)
    if(ARM_COMPUTE_TOOLCHAIN_PREFIX)
        ov_arm_compute_add_option("toolchain_prefix" "${ARM_COMPUTE_TOOLCHAIN_PREFIX}")
    endif()

    # ==========================================================================
    # Platform-specific Configuration
    # ==========================================================================

    # Initialize flags
    set(extra_cxx_flags "${CMAKE_CXX_FLAGS} -Wno-undef")
    if(MSVC64)
        string(REPLACE "/MP " "" extra_cxx_flags "${extra_cxx_flags}")
    elseif(CMAKE_POSITION_INDEPENDENT_CODE)
        set(extra_cxx_flags "${extra_cxx_flags} -fPIC")
    endif()

    # Configure platform
    ov_arm_compute_configure_platform()

    # Configure compiler flags
    ov_arm_compute_configure_flags()

    # Add extra flags to options
    if(extra_link_flags)
        ov_arm_compute_add_option("extra_link_flags" "${extra_link_flags}")
    endif()
    if(extra_cxx_flags)
        ov_arm_compute_add_option("extra_cxx_flags" "${extra_cxx_flags}")
    endif()
    if(extra_cc_flags)
        ov_arm_compute_add_option("extra_cc_flags" "${extra_cc_flags}")
    endif()

    # Verbosity
    if(NOT CMAKE_VERBOSE_MAKEFILE)
        list(APPEND ARM_COMPUTE_OPTIONS "--silent")
    endif()

    # Fixed format kernels
    ov_arm_compute_add_option("fixed_format_kernels" "True")

    # ==========================================================================
    # Build Configuration
    # ==========================================================================

    # Set output library name
    if(MSVC64)
        set(arm_compute build/${OV_CPU_ARM_TARGET_ARCH}/arm_compute-static.lib)
    else()
        set(arm_compute build/${OV_CPU_ARM_TARGET_ARCH}/libarm_compute-static.a)
    endif()

    # Configure and build ACL using ExternalProject
    ExternalProject_Add(arm_compute_build
        PREFIX ${ARM_COMPUTE_BUILD_DIR}
        SOURCE_DIR ${ARM_COMPUTE_SOURCE_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E env ${cmake_build_env}
            ${SCONS} ${ARM_COMPUTE_OPTIONS} ${arm_compute}
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${ARM_COMPUTE_BUILD_DIR}/${arm_compute}
        INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory ${ARM_COMPUTE_BUILD_DIR}/build/${OV_CPU_ARM_TARGET_ARCH}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${ARM_COMPUTE_SOURCE_DIR}/build/${OV_CPU_ARM_TARGET_ARCH}/libarm_compute-static.a
                ${ARM_COMPUTE_BUILD_DIR}/build/${OV_CPU_ARM_TARGET_ARCH}/libarm_compute-static.a
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${ARM_COMPUTE_SOURCE_DIR}/build
        LOG_BUILD ${ENABLE_DEBUG_CAPS}
    )

    # Get the full path to the built library
    set(arm_compute_full_path "${ARM_COMPUTE_BUILD_DIR}/${arm_compute}")

    # Debug messages
    message(DEBUG "ACL build configuration:")
    message(DEBUG "  ARM_COMPUTE_BUILD_DIR: ${ARM_COMPUTE_BUILD_DIR}")
    message(DEBUG "  arm_compute: ${arm_compute}")
    message(DEBUG "  arm_compute_full_path: ${arm_compute_full_path}")

    # ==========================================================================
    # Create Imported Targets
    # ==========================================================================

    find_package(Threads REQUIRED)

    # Create imported targets
    add_library(arm_compute::arm_compute STATIC IMPORTED GLOBAL)
    set_target_properties(arm_compute::arm_compute PROPERTIES
        IMPORTED_LOCATION ${arm_compute_full_path}
        INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_SOURCE_DIR}
        INTERFACE_LINK_LIBRARIES Threads::Threads
        OSX_ARCHITECTURES arm64)
    add_dependencies(arm_compute::arm_compute arm_compute_build)

    add_library(arm_compute::half INTERFACE IMPORTED GLOBAL)
    set_target_properties(arm_compute::half PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_SOURCE_DIR}/include)

    # ==========================================================================
    # Setup for oneDNN Integration
    # ==========================================================================

    set(ACL_FOUND ON)
    set(ACL_LIBRARIES arm_compute::arm_compute arm_compute::half)

    foreach(acl_library IN LISTS ACL_LIBRARIES)
        list(APPEND ACL_INCLUDE_DIRS
                $<TARGET_PROPERTY:${acl_library},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()

    # Required by oneDNN to parse ACL version
    set(ACL_INCLUDE_DIR "${ARM_COMPUTE_SOURCE_DIR}")
    set(ENV{ACL_ROOT_DIR} "${ARM_COMPUTE_SOURCE_DIR}")
endif()

