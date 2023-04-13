# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ARM_COMPUTE_INCLUDE_DIR OR ARM_COMPUTE_LIB_DIR)
    if (NOT ARM_COMPUTE_INCLUDE_DIR)
        message(FATAL_ERROR "Undefined ARM_COMPUTE_INCLUDE_DIR input variable should be set manually")
    else()
        message(STATUS "Using ${ARM_COMPUTE_INCLUDE_DIR} to include arm compute library headers")
    endif()

    if (NOT ARM_COMPUTE_LIB_DIR)
        message(FATAL_ERROR "Undefined ARM_COMPUTE_LIB_DIR input variable should be set manually")
    else()
        find_library(
            ARM_COMPUTE_LIB
            arm_compute-static
            PATHS ${ARM_COMPUTE_LIB_DIR}
        )
        message(STATUS "Found arm_compute-static: ${ARM_COMPUTE_LIB}")
        add_library(arm_compute STATIC IMPORTED GLOBAL)
        set_target_properties(arm_compute PROPERTIES
            IMPORTED_LOCATION ${ARM_COMPUTE_LIB}
            INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_INCLUDE_DIR})
    endif()

    add_library(half INTERFACE IMPORTED GLOBAL)
    set_target_properties(half PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_INCLUDE_DIR})
else()
    set(ARM_COMPUTE_SOURCE_DIR ${intel_cpu_thirdparty_SOURCE_DIR}/ComputeLibrary)
    set(ARM_COMPUTE_BINARY_DIR ${intel_cpu_thirdparty_BINARY_DIR}/ComputeLibrary)

    message(STATUS "Configure to build ${ARM_COMPUTE_SOURCE_DIR}")

    find_host_program(SCONS scons)

    if(NOT SCONS)
        message(FATAL_ERROR "Scons tool is not found!")
    endif()

    file(GLOB_RECURSE SOURCES
        ${ARM_COMPUTE_SOURCE_DIR}/*.cpp
        ${ARM_COMPUTE_SOURCE_DIR}/*.hpp
        ${ARM_COMPUTE_SOURCE_DIR}/*.h
    )

    set(extra_cxx_flags "${CMAKE_CXX_FLAGS} -Wno-undef")
    if(MSVC64)
        # clang-cl does not recognize /MP option
        string(REPLACE "/MP " "" extra_cxx_flags "${extra_cxx_flags}")
    else()
        # -fPIC is not applicable for clang-cl
        set(extra_cxx_flags "${extra_cxx_flags} -fPIC")
    endif()

    set(ARM_COMPUTE_OPTIONS
        neon=1
        opencl=0
        openmp=0
        cppthreads=1
        examples=0
        Werror=0
        gemm_tuner=0
        reference_openmp=0
        validation_tests=0
        benchmark_tests=0
        # TODO: check this for Apple Silicon
        # multi_isa=1
        # TODO: use CC for ARM compute library to minimize binary size
        # build_config=<file>
        # TODO: use data_type_support to disable useless kernels
        data_layout_support=all
        arch=${ARM_COMPUTE_TARGET_ARCH}
    )

    if(NOT MSVC64)
        list(APPEND ARM_COMPUTE_OPTIONS
            build_dir=${ARM_COMPUTE_BINARY_DIR}
            install_dir=${ARM_COMPUTE_BINARY_DIR}/install)
    endif()

    if(ARM_COMPUTE_SCONS_JOBS)
        list(APPEND ARM_COMPUTE_OPTIONS --jobs=${ARM_COMPUTE_SCONS_JOBS})
    endif()

    set(ARM_COMPUTE_DEBUG_OPTIONS
        debug=1
        asserts=1
        logging=1)

    # cmake older 3.20 does not support generator expressions in add_custom_command
    # https://cmake.org/cmake/help/latest/command/add_custom_command.html#examples-generating-files
    if(OV_GENERATOR_MULTI_CONFIG AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
        foreach(option IN LISTS ARM_COMPUTE_DEBUG_OPTIONS)
            list(APPEND ARM_COMPUTE_OPTIONS $<$<CONFIG:Debug>:${option}>)
        endforeach()
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND ARM_COMPUTE_OPTIONS ${ARM_COMPUTE_DEBUG_OPTIONS})
    endif()

    if(EMSCRIPTEN OR LINUX)
        list(APPEND ARM_COMPUTE_OPTIONS os=linux)
    elseif(ANDROID)
        list(APPEND ARM_COMPUTE_OPTIONS os=android)
    elseif(APPLE)
        list(APPEND ARM_COMPUTE_OPTIONS os=macos)
    elseif(WIN32)
        list(APPEND ARM_COMPUTE_OPTIONS os=windows)
    endif()

    if(CMAKE_CROSSCOMPILING)
        list(APPEND ARM_COMPUTE_OPTIONS build=cross_compile)
    else()
        list(APPEND ARM_COMPUTE_OPTIONS build=native)
    endif()

    if (CMAKE_CXX_COMPILER_LAUNCHER)
        list(APPEND ARM_COMPUTE_OPTIONS compiler_cache=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()

    # used to build for yocto
    if(ARM_COMPUTE_TOOLCHAIN_PREFIX)
        list(APPEND ARM_COMPUTE_OPTIONS toolchain_prefix=${ARM_COMPUTE_TOOLCHAIN_PREFIX})
    endif()

    if(ANDROID)
        if(ANDROID_PLATFORM_LEVEL LESS 18)
            message(FATAL_ERROR "ARM compute library requires Android API 18 level and higher"
                                "Please, speficy -DANDROID_PLATFORM=android-18 at least")
        endif()

        if(ANDROID_NDK_REVISION LESS "23.0")
            list(APPEND ARM_COMPUTE_OPTIONS toolchain_prefix="${ANDROID_TOOLCHAIN_PREFIX}")
        else()
            string(REGEX REPLACE "/bin/[^/]+-" "/bin/llvm-" ANDROID_TOOLCHAIN_PREFIX_FIXED "${ANDROID_TOOLCHAIN_PREFIX}")
            message(STATUS "SCONS: using ANDROID_TOOLCHAIN_PREFIX=${ANDROID_TOOLCHAIN_PREFIX_FIXED}")
            list(APPEND ARM_COMPUTE_OPTIONS toolchain_prefix="${ANDROID_TOOLCHAIN_PREFIX_FIXED}")
        endif()

        list(APPEND ARM_COMPUTE_OPTIONS
            compiler_prefix="${ANDROID_TOOLCHAIN_ROOT}/bin/")

        set(extra_flags "${extra_flags} --target=${ANDROID_LLVM_TRIPLE}")
        set(extra_flags "${extra_flags} --gcc-toolchain=${ANDROID_TOOLCHAIN_ROOT}")
        set(extra_flags "${extra_flags} --sysroot=${CMAKE_SYSROOT}")

        set(extra_link_flags "${extra_link_flags} ${extra_flags}")
        set(extra_cxx_flags "${extra_cxx_flags} ${extra_flags}")
    elseif(CMAKE_CROSSCOMPILING AND LINUX)
        get_filename_component(cxx_compiler "${CMAKE_CXX_COMPILER}" NAME)
        get_filename_component(c_compiler "${CMAKE_C_COMPILER}" NAME)
        get_filename_component(compiler_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)

        set(cmake_build_env
            CC=${c_compiler}
            CXX=${cxx_compiler})
        list(APPEND ARM_COMPUTE_OPTIONS compiler_prefix="${compiler_prefix}/")
    elseif(EMSCRIPTEN)
        set(cmake_build_env
            CC=emcc
            CXX=em++
            RANLIB=emranlib
            AR=emar)
        # EMSDK: Passing any of -msse, -msse2, -msse3, -mssse3, -msse4.1, -msse4.2,
        # -msse4, -mavx, -mfpu=neon flags also requires passing -msimd128 (or -mrelaxed-simd)!
        set(extra_cxx_flags "${extra_cxx_flags} -msimd128")
        # clang-16: error: argument unused during compilation: '-mthumb' [-Werror,-Wunused-command-line-argument]
        # clang-16: error: argument unused during compilation: '-mfloat-abi=hard' [-Werror,-Wunused-command-line-argument]
        set(extra_cxx_flags "${extra_cxx_flags} \
            -Wno-unused-command-line-argument \
            -Wno-unknown-warning-option \
            -Wno-unused-function \
            -Wno-unused-but-set-variable")

        get_filename_component(toolchain_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)
        list(APPEND ARM_COMPUTE_OPTIONS toolchain_prefix="${toolchain_prefix}/")
    elseif(APPLE)
        if(CMAKE_OSX_DEPLOYMENT_TARGET)
            set(extra_cxx_flags "${extra_cxx_flags} -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
            set(minos_added ON)
        endif()

        if(HOST_X86_64)
            if(NOT minos_added)
                message(FATAL_ERROR "Please, specify either env MACOSX_DEPLOYMENT_TARGET or cmake CMAKE_OSX_DEPLOYMENT_TARGET variables")
            endif()
            set(extra_cxx_flags "${extra_cxx_flags} --sysroot ${CMAKE_OSX_SYSROOT}")
        endif()

        set(extra_cxx_flags "${extra_cxx_flags} -Wno-error=return-stack-address")
        get_filename_component(compiler_prefix "${CMAKE_CXX_COMPILER}" DIRECTORY)
        list(APPEND ARM_COMPUTE_OPTIONS compiler_prefix="${compiler_prefix}/")

        if(CMAKE_OSX_ARCHITECTURES)
            foreach(arch IN LISTS CMAKE_OSX_ARCHITECTURES)
                set(extra_cxx_flags "${extra_cxx_flags} -arch ${arch}")
            endforeach()
        endif()
    elseif(MSVC64)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
            set(extra_cxx_flags "${extra_cxx_flags} $<IF:$<CONFIG:Release>,/MD,/MDd>")
        else()
            if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                set(extra_cxx_flags "${extra_cxx_flags} /MDd")
            else()
                set(extra_cxx_flags "${extra_cxx_flags} /MD")
            endif()
        endif()
    endif()

    if(ENABLE_LTO)
        if((CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG) AND (NOT CMAKE_CROSSCOMPILING))
            set(extra_cxx_flags "${extra_cxx_flags} -flto=thin")
            set(extra_link_flags "${extra_link_flags} -flto=thin")
        endif()
    endif()

    if(SUGGEST_OVERRIDE_SUPPORTED)
        set(extra_cxx_flags "${extra_cxx_flags} -Wno-suggest-override")
    endif()

    if(extra_link_flags)
        list(APPEND ARM_COMPUTE_OPTIONS extra_link_flags=${extra_link_flags})
    endif()
    if(extra_cxx_flags)
        list(APPEND ARM_COMPUTE_OPTIONS extra_cxx_flags=${extra_cxx_flags})
    endif()

    if(NOT CMAKE_VERBOSE_MAKEFILE)
        list(APPEND ARM_COMPUTE_OPTIONS --silent)
    endif()

    if(MSVC64)
        set(arm_compute build/arm_compute-static.lib)
        set(arm_compute_full_path "${ARM_COMPUTE_SOURCE_DIR}/${arm_compute}")
    else()
        set(arm_compute ${ARM_COMPUTE_BINARY_DIR}/libarm_compute-static.a)
        set(arm_compute_full_path "${arm_compute}")
    endif()

    add_custom_command(
        OUTPUT
            ${arm_compute_full_path}
        COMMAND ${CMAKE_COMMAND} -E env ${cmake_build_env}
            ${SCONS} ${ARM_COMPUTE_OPTIONS}
                ${arm_compute}
        WORKING_DIRECTORY ${ARM_COMPUTE_SOURCE_DIR}
        COMMENT "Build Arm Compute Library"
        DEPENDS ${SOURCES}
                ${CMAKE_CURRENT_LIST_FILE}
                ${ARM_COMPUTE_SOURCE_DIR}/SConscript
                ${ARM_COMPUTE_SOURCE_DIR}/SConstruct)

    # Import targets

    add_custom_target(arm_compute_static_libs DEPENDS ${arm_compute_full_path})

    add_library(arm_compute::arm_compute STATIC IMPORTED GLOBAL)
    set_target_properties(arm_compute::arm_compute PROPERTIES
        IMPORTED_LOCATION ${arm_compute_full_path}
        INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_SOURCE_DIR})
    add_dependencies(arm_compute::arm_compute arm_compute_static_libs)

    add_library(arm_compute::half INTERFACE IMPORTED GLOBAL)
    set_target_properties(arm_compute::half PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${ARM_COMPUTE_SOURCE_DIR}/include)

    # Compute Library uses cppthreads=1
    # if one day will rely on TBB only, we can omit this dependency
    find_package(Threads REQUIRED)
    set_target_properties(arm_compute::arm_compute PROPERTIES
        INTERFACE_LINK_LIBRARIES Threads::Threads)

    set(ACL_FOUND ON)
    set(ACL_LIBRARIES arm_compute::arm_compute arm_compute::half)

    foreach(acl_library IN LISTS ACL_LIBRARIES)
        list(APPEND ACL_INCLUDE_DIRS
                $<TARGET_PROPERTY:${acl_library},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()

    # required by oneDNN to attempt to parse ACL version
    set(ENV{ACL_ROOT_DIR} "${ARM_COMPUTE_SOURCE_DIR}")
endif()
