# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (target_flags)
include (options)

# these options are aimed to optimize build time on development system

if(X86_64)
    set(ENABLE_MKL_DNN_DEFAULT ON)
else()
    set(ENABLE_MKL_DNN_DEFAULT OFF)
endif()

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ${ENABLE_MKL_DNN_DEFAULT})

ie_dependent_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON "X86_64;NOT APPLE;NOT MINGW;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)

# FIXME: there are compiler failures with LTO and Cross-Compile toolchains. Disabling for now, but
#        this must be addressed in a proper way
ie_dependent_option (ENABLE_LTO "Enable Link Time Optimization" OFF "LINUX;NOT CMAKE_CROSSCOMPILING; CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9" OFF)

ie_option (OS_FOLDER "create OS dedicated folder in output" OFF)

# FIXME: ARM cross-compiler generates several "false positive" warnings regarding __builtin_memcpy buffer overflow
ie_dependent_option (TREAT_WARNING_AS_ERROR "Treat build warnings as errors" ON "X86 OR X86_64" OFF)

ie_option (ENABLE_INTEGRITYCHECK "build DLLs with /INTEGRITYCHECK flag" OFF)

ie_option (ENABLE_SANITIZER "enable checking memory errors via AddressSanitizer" OFF)

ie_option (ENABLE_THREAD_SANITIZER "enable checking data races via ThreadSanitizer" OFF)

ie_dependent_option (COVERAGE "enable code coverage" OFF "CMAKE_CXX_COMPILER_ID STREQUAL GNU" OFF)

# Define CPU capabilities

ie_dependent_option (ENABLE_SSE42 "Enable SSE4.2 optimizations" ON "X86_64 OR X86" OFF)

ie_dependent_option (ENABLE_AVX2 "Enable AVX2 optimizations" ON "X86_64 OR X86" OFF)

ie_dependent_option (ENABLE_AVX512F "Enable AVX512 optimizations" ON "X86_64 OR X86" OFF)

ie_option (ENABLE_PROFILING_ITT "Build with ITT tracing. Optionally configure pre-built ittnotify library though INTEL_VTUNE_DIR variable." OFF)

ie_option (ENABLE_DOCS "Build docs using Doxygen" OFF)

ie_dependent_option (ENABLE_FASTER_BUILD "Enable build features (PCH, UNITY) to speed up build time" OFF "CMAKE_VERSION VERSION_GREATER_EQUAL 3.16" OFF)

# Type of build, we add this as an explicit option to default it to ON
# FIXME: Ah this moment setting this to OFF will only build ngraph a static library
ie_option (BUILD_SHARED_LIBS "Build as a shared library" ON)

ie_dependent_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ON "UNIX;NOT ANDROID" OFF)

ie_dependent_option(ENABLE_CPPLINT_REPORT "Build cpplint report instead of failing the build" OFF "ENABLE_CPPLINT" OFF)

ie_option(ENABLE_CLANG_FORMAT "Enable clang-format checks during the build" ON)

ie_option_enum(SELECTIVE_BUILD "Enable OpenVINO conditional compilation or statistics collection. \
In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected InelSEAPI statistics. \
Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv" OFF
               ALLOWED_VALUES ON OFF COLLECT)
