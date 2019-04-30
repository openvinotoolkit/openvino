# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

include (options)

#this options are aimed to optimize build time on development system

#backed targets

ie_option (ENABLE_GNA "GNA support for inference engine" ON)

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON)

ie_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON)

ie_option (ENABLE_PROFILING_ITT "ITT tracing of IE and plugins internals" ON)

ie_option (ENABLE_PROFILING_RAW "Raw counters profiling (just values, no start/stop time or timeline)" OFF)

# "MKL-DNN library might use MKL-ML or OpenBLAS for gemm tasks: MKL|OPENBLAS|JIT"
if (NOT GEMM STREQUAL "MKL"
        AND NOT GEMM STREQUAL "OPENBLAS"
        AND NOT GEMM STREQUAL "JIT")
    set (GEMM "JIT")
    message(STATUS "GEMM should be set to MKL, OPENBLAS or JIT. Default option is " ${GEMM})
endif()
set(GEMM "${GEMM}" CACHE STRING "Gemm implementation" FORCE)
list (APPEND IE_OPTIONS GEMM)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if (NOT THREADING STREQUAL "TBB"
        AND NOT THREADING STREQUAL "OMP"
        AND NOT THREADING STREQUAL "SEQ")
    set (THREADING "TBB")
    message(STATUS "THREADING should be set to TBB, OMP or SEQ. Default option is " ${THREADING})
endif()
set(THREADING "${THREADING}" CACHE STRING "Threading" FORCE)
list (APPEND IE_OPTIONS THREADING)

# Enable postfixes for Debug/Release builds
set (IE_DEBUG_POSTFIX_WIN "d")
set (IE_RELEASE_POSTFIX_WIN "")
set (IE_DEBUG_POSTFIX_LIN "")
set (IE_RELEASE_POSTFIX_LIN "")
set (IE_DEBUG_POSTFIX_MAC "d")
set (IE_RELEASE_POSTFIX_MAC "")

if (WIN32)
    set (IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_WIN})
    set (IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_WIN})
elseif(APPLE)
    set (IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_MAC})
    set (IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_MAC})
else()
    set (IE_DEBUG_POSTFIX ${IE_DEBUG_POSTFIX_LIN})
    set (IE_RELEASE_POSTFIX ${IE_RELEASE_POSTFIX_LIN})
endif()
set(IE_DEBUG_POSTFIX "${IE_DEBUG_POSTFIX}" CACHE STRING "Debug postfix" FORCE)
list (APPEND IE_OPTIONS IE_DEBUG_POSTFIX)
set(IE_RELEASE_POSTFIX "${IE_RELEASE_POSTFIX}" CACHE STRING "Release postfix" FORCE)
list (APPEND IE_OPTIONS IE_RELEASE_POSTFIX)

ie_option (ENABLE_TESTS "unit and functional tests" OFF)

ie_option (ENABLE_GAPI_TESTS "unit tests for GAPI kernels" OFF)

ie_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF)

ie_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON)

ie_option (ENABLE_SAMPLES_CORE "console samples core library" ON)

ie_option (ENABLE_SANITIZER "enable checking memory errors via AddressSanitizer" OFF)

ie_option (COVERAGE "enable code coverage" OFF)

ie_option (ENABLE_STRESS_UNIT_TESTS "stress unit tests" OFF)

ie_option (VERBOSE_BUILD "shows extra information about build" OFF)

ie_option (ENABLE_UNSAFE_LOCATIONS "skip check for MD5 for dependency" OFF)

ie_option (ENABLE_ALTERNATIVE_TEMP "in case of dependency conflict, to avoid modification in master, use local copy of dependency" ON)

ie_option (ENABLE_SEGMENTATION_TESTS "segmentation tests" ON)

ie_option (ENABLE_OBJECT_DETECTION_TESTS "object detection tests" ON)

ie_option (ENABLE_OPENCV "enables OpenCV" ON)

ie_option (OS_FOLDER "create OS dedicated folder in output" OFF)

ie_option (ENABLE_PLUGIN_RPATH "enables rpath information to be present in plugins binary, and in corresponding test_applications" ON)

ie_option (ENABLE_AFFINITY_GENERATOR "enables affinity generator build" OFF)

ie_option (ENABLE_DEBUG_SYMBOLS "generates symbols for debugging" OFF)

ie_option (ENABLE_PYTHON "enables ie python bridge build" OFF)

ie_option (TREAT_WARNING_AS_ERROR "Treat build warnings as errors" ON)

ie_option(ENABLE_CPPLINT "Enable cpplint checks during the build" OFF)
ie_option(ENABLE_CPPLINT_REPORT "Build cpplint report instead of failing the build" OFF)

#environment variables used

#name of environment variable stored path to temp directory"
set (DL_SDK_TEMP  "DL_SDK_TEMP")
