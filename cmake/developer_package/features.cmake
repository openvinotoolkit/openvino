# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(options)
include(target_flags)

# FIXME: there are compiler failures with LTO and Cross-Compile toolchains. Disabling for now, but
#        this must be addressed in a proper way
ie_dependent_option (ENABLE_LTO "Enable Link Time Optimization" OFF "LINUX;NOT CMAKE_CROSSCOMPILING; CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9" OFF)

ie_option (OS_FOLDER "create OS dedicated folder in output" OFF)

if(UNIX)
    ie_option(USE_BUILD_TYPE_SUBFOLDER "Create dedicated sub-folder per build type for output binaries" ON)
else()
    ie_option(USE_BUILD_TYPE_SUBFOLDER "Create dedicated sub-folder per build type for output binaries" OFF)
endif()

# FIXME: ARM cross-compiler generates several "false positive" warnings regarding __builtin_memcpy buffer overflow
ie_dependent_option (TREAT_WARNING_AS_ERROR "Treat build warnings as errors" ON "X86 OR X86_64" OFF)

ie_option (ENABLE_INTEGRITYCHECK "build DLLs with /INTEGRITYCHECK flag" OFF)

ie_option (ENABLE_SANITIZER "enable checking memory errors via AddressSanitizer" OFF)

ie_option (ENABLE_THREAD_SANITIZER "enable checking data races via ThreadSanitizer" OFF)

ie_dependent_option (ENABLE_COVERAGE "enable code coverage" OFF "CMAKE_CXX_COMPILER_ID STREQUAL GNU" OFF)

# Defines CPU capabilities

ie_dependent_option (ENABLE_SSE42 "Enable SSE4.2 optimizations" ON "X86_64 OR X86" OFF)

ie_dependent_option (ENABLE_AVX2 "Enable AVX2 optimizations" ON "X86_64 OR X86" OFF)

ie_dependent_option (ENABLE_AVX512F "Enable AVX512 optimizations" ON "X86_64 OR X86" OFF)

# Type of build, we add this as an explicit option to default it to ON
# FIXME: Ah this moment setting this to OFF will only build ngraph a static library
ie_option (BUILD_SHARED_LIBS "Build as a shared library" ON)

ie_dependent_option (ENABLE_FASTER_BUILD "Enable build features (PCH, UNITY) to speed up build time" OFF "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)

if(NOT DEFINED ENABLE_CPPLINT)
	ie_dependent_option (ENABLE_CPPLINT "Enable cpplint checks during the build" ON "UNIX;NOT ANDROID" OFF)
endif()

if(NOT DEFINED ENABLE_CPPLINT_REPORT)
	ie_dependent_option (ENABLE_CPPLINT_REPORT "Build cpplint report instead of failing the build" OFF "ENABLE_CPPLINT" OFF)
endif()

ie_dependent_option (ENABLE_CLANG_FORMAT "Enable clang-format checks during the build" ON "UNIX;NOT ANDROID" OFF)

ie_option (VERBOSE_BUILD "shows extra information about build" OFF)

ie_option (ENABLE_UNSAFE_LOCATIONS "skip check for MD5 for dependency" OFF)

ie_dependent_option (ENABLE_FUZZING "instrument build for fuzzing" OFF "CMAKE_CXX_COMPILER_ID MATCHES ^(Apple)?Clang$; NOT WIN32" OFF)

#
# Check features
#

if(ENABLE_AVX512F)
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") AND (MSVC_VERSION VERSION_LESS 1920))
        # 1920 version of MSVC 2019. In MSVC 2017 AVX512F not work
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6))
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10))
        # TBD: clarify which AppleClang version supports avx512
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
    if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9))
        set(ENABLE_AVX512F OFF CACHE BOOL "" FORCE)
    endif()
endif()

if (VERBOSE_BUILD)
    set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "" FORCE)
endif()
