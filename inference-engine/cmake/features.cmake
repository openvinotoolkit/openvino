# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (options)

#these options are aimed to optimize build time on development system

#backed targets
ie_option (ENABLE_GNA "GNA support for inference engine" OFF)
ie_option (ENABLE_ROCKHOPER "use Rockhopper decoder for converting / output scores" OFF)

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON)

ie_option (ENABLE_CLDNN "clDnn based plugin for inference engine" OFF)

ie_option (ENABLE_CLDNN_TESTS "Enable clDNN unit tests" OFF)

ie_option (ENABLE_CLDNN_BUILD "build clDnn from sources" OFF)

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
        AND NOT THREADING STREQUAL "TBB_AUTO"
        AND NOT THREADING STREQUAL "OMP"
        AND NOT THREADING STREQUAL "SEQ")
    set (THREADING "OMP")
    message(STATUS "THREADING should be set to TBB, TBB_AUTO, OMP or SEQ. Default option is " ${THREADING})
endif()
set(THREADING "${THREADING}" CACHE STRING "Threading" FORCE)
list (APPEND IE_OPTIONS THREADING)

ie_option (ENABLE_VPU "vpu targeted plugins for inference engine" OFF)

ie_option (ENABLE_MYRIAD "myriad targeted plugin for inference engine" OFF)

ie_option (ENABLE_MYRIAD_NO_BOOT "myriad plugin will skip device boot" OFF)

ie_option (ENABLE_TESTS "unit and functional tests" OFF)

ie_option (ENABLE_GAPI_TESTS "tests for GAPI kernels" OFF)

ie_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF)

ie_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON)

ie_option (ENABLE_SAMPLES_CORE "console samples core library" ON)

ie_option (ENABLE_SANITIZER "enable checking memory errors via AddressSanitizer" OFF)

ie_option (ENABLE_FUZZING "instrument build for fuzzing" OFF)

ie_option (COVERAGE "enable code coverage" OFF)

ie_option (VERBOSE_BUILD "shows extra information about build" OFF)

ie_option (ENABLE_UNSAFE_LOCATIONS "skip check for MD5 for dependency" OFF)

ie_option (ENABLE_ALTERNATIVE_TEMP "in case of dependency conflict, to avoid modification in master, use local copy of dependency" ON)

ie_option (ENABLE_SEGMENTATION_TESTS "segmentation tests" ON)

ie_option (ENABLE_OBJECT_DETECTION_TESTS "object detection tests" ON)

ie_option (ENABLE_DUMP "enables mode for dumping per layer information" OFF)

ie_option (ENABLE_OPENCV "enables OpenCV" OFF)

ie_option (OS_FOLDER "create OS dedicated folder in output" OFF)

ie_option (ENABLE_PLUGIN_RPATH "enables rpath information to be present in plugins binary, and in corresponding test_applications" ON)

ie_option (ENABLE_AFFINITY_GENERATOR "enables affinity generator build" OFF)

ie_option (ENABLE_DEBUG_SYMBOLS "generates symbols for debugging" OFF)

ie_option (ENABLE_PYTHON "enables ie python bridge build" OFF)

ie_option (DEVELOPMENT_PLUGIN_MODE "Disabled build of all plugins" OFF)

ie_option (TREAT_WARNING_AS_ERROR "Treat build warnings as errors" ON)

ie_option (ENABLE_CPP_CCT "enables C++ version of Cross Check Tool" OFF)

ie_option (ENABLE_UNICODE_PATH_SUPPORT "Enable loading models from Unicode paths" ON)

ie_option (ENABLE_LTO "Enable Link Time Optimization" OFF)

# FIXME: there are compiler failures with LTO and Cross-Compile toolchains. Disabling for now, but
#        this must be addressed in a proper way
if(CMAKE_CROSSCOMPILING OR NOT (UNIX AND NOT APPLE))
    set(ENABLE_LTO OFF)
endif()

if (UNIX AND NOT APPLE AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
    set(ENABLE_UNICODE_PATH_SUPPORT OFF)
endif()

if (UNIX AND NOT APPLE)
    ie_option(ENABLE_CPPLINT "Enable cpplint checks during the build" ON)
    ie_option(ENABLE_CPPLINT_REPORT "Build cpplint report instead of failing the build" OFF)
else()
    set(ENABLE_CPPLINT OFF)
endif()

if (UNIX AND NOT APPLE AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.10)
    ie_option(ENABLE_CPPCHECK "Enable cppcheck during the build" OFF)
else()
    set(ENABLE_CPPCHECK OFF)
endif()

#environment variables used

#name of environment variable stored path to temp directory"
set (DL_SDK_TEMP  "DL_SDK_TEMP")
