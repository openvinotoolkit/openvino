# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (target_flags)
include (options)

#these options are aimed to optimize build time on development system

ie_option (ENABLE_GNA "GNA support for inference engine" ON)

ie_option (ENABLE_CLDNN_TESTS "Enable clDNN unit tests" OFF)

ie_option (ENABLE_PROFILING_ITT "ITT tracing of IE and plugins internals" ON)

ie_option (ENABLE_PROFILING_RAW "Raw counters profiling (just values, no start/stop time or timeline)" OFF)

# "MKL-DNN library might use MKL-ML or OpenBLAS for gemm tasks: MKL|OPENBLAS|JIT"
if (NOT GEMM STREQUAL "MKL"
        AND NOT GEMM STREQUAL "OPENBLAS"
        AND NOT GEMM STREQUAL "JIT")
    if(ANDROID)
        set(GEMM "JIT")
    else()
        set(GEMM "JIT")
    endif()
    message(STATUS "GEMM should be set to MKL, OPENBLAS or JIT. Default option is " ${GEMM})
endif()
set(GEMM "${GEMM}" CACHE STRING "Gemm implementation" FORCE)
list (APPEND IE_OPTIONS GEMM)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if (NOT THREADING STREQUAL "TBB"
        AND NOT THREADING STREQUAL "TBB_AUTO"
        AND NOT THREADING STREQUAL "OMP"
        AND NOT THREADING STREQUAL "SEQ")
    if (ARM OR AARCH64)
        set (THREADING "SEQ")
    else()
        set (THREADING "TBB")
    endif()
    message(STATUS "THREADING should be set to TBB, TBB_AUTO, OMP or SEQ. Default option is " ${THREADING})
endif()
set(THREADING "${THREADING}" CACHE STRING "Threading" FORCE)
list (APPEND IE_OPTIONS THREADING)

ie_option (ENABLE_VPU "vpu targeted plugins for inference engine" ON)

ie_option (ENABLE_MYRIAD "myriad targeted plugin for inference engine" ON)

ie_option (ENABLE_MYRIAD_NO_BOOT "myriad plugin will skip device boot" OFF)

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_option (ENABLE_GAPI_TESTS "tests for GAPI kernels" OFF)

ie_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF)

ie_option (ENABLE_MYRIAD_MVNC_TESTS "functional and behavior tests for mvnc api" OFF)

ie_option (ENABLE_BEH_TESTS "tests oriented to check inference engine API corecteness" ON)

ie_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON)

ie_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON)

ie_option (ENABLE_FUZZING "instrument build for fuzzing" OFF)

ie_option (COVERAGE "enable code coverage" OFF)

ie_option (VERBOSE_BUILD "shows extra information about build" OFF)

ie_option (ENABLE_UNSAFE_LOCATIONS "skip check for MD5 for dependency" OFF)

ie_option (ENABLE_ALTERNATIVE_TEMP "in case of dependency conflict, to avoid modification in master, use local copy of dependency" ON)

ie_option (ENABLE_DUMP "enables mode for dumping per layer information" OFF)

ie_option (ENABLE_OPENCV "enables OpenCV" ON)

ie_option (ENABLE_DEBUG_SYMBOLS "generates symbols for debugging" OFF)

ie_option (ENABLE_PYTHON "enables ie python bridge build" OFF)

ie_option (ENABLE_CPP_CCT "enables C++ version of Cross Check Tool" OFF)

ie_option (ENABLE_UNICODE_PATH_SUPPORT "Enable loading models from Unicode paths" ON)

ie_option (ENABLE_IR_READER "Compile with IR readers / parsers" ON)

ie_option (ENABLE_C "enables ie c bridge build" ON)

ie_option(ENABLE_CPPLINT "Enable cpplint checks during the build" OFF)

ie_option(ENABLE_CPPLINT_REPORT "Build cpplint report instead of failing the build" OFF)

ie_option(ENABLE_CLANG_FORMAT "Enable clang-format checks during the build" OFF)

ie_option(ENABLE_CPPCHECK "Enable cppcheck during the build" OFF)

set(IE_EXTRA_PLUGINS "" CACHE STRING "Extra paths for plugins to include into DLDT build tree")

if (LINUX)
    ie_option(ENABLE_TBB_RELEASE_ONLY "Only Release TBB libraries are linked to the Inference Engine binaries" ON)
endif()