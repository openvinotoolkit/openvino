# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required (VERSION 2.8)

include ("options")

#this options are aimed to optimize build time on development system

#backed targets

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON)

ie_option (ENABLE_MKL_DNN_JIT "Enable JIT for the MKL DNN plugin" ON)

ie_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON)

ie_option (ENABLE_PROFILING_ITT "ITT tracing of IE and plugins internals" ON)

ie_option (ENABLE_PROFILING_RAW "Raw counters profiling (just values, no start/stop time or timeline)" OFF)

#

# "MKL-DNN library might use MKL-ML or OpenBLAS for gemm tasks: MKL|OPENBLAS|JIT"
if (NOT GEMM STREQUAL "MKL" AND NOT GEMM STREQUAL "OPENBLAS" AND NOT GEMM STREQUAL "JIT")
    set (GEMM "JIT")
    message(STATUS "GEMM should be set to MKL|OPENBLAS|JIT. Default option is " ${GEMM})
endif()
list (APPEND IE_OPTIONS GEMM)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if (NOT THREADING STREQUAL "TBB" AND NOT THREADING STREQUAL "OMP" AND NOT THREADING STREQUAL "SEQ")
    set (THREADING "OMP")
    message(STATUS "THREADING should be set to TBB|OMP|SEQ. Default option is " ${THREADING})
endif()
list (APPEND IE_OPTIONS THREADING)

ie_option (ENABLE_INTEL_OMP "MKL-DNN library based on Intel OMP implementation" ON)

ie_option (ENABLE_TESTS "unit and functional tests" OFF)

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

#environment variables used

#name of environment variable stored path to temp directory"
set (DL_SDK_TEMP  "DL_SDK_TEMP")
