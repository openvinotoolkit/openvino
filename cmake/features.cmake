# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common cmake options
#

ie_dependent_option (ENABLE_INTEL_CPU "CPU plugin for OpenVINO Runtime" ON "RISCV64 OR X86 OR X86_64" OFF)

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_option (ENABLE_COMPILE_TOOL "Enables compile_tool" ON)

ie_option (ENABLE_STRICT_DEPENDENCIES "Skip configuring \"convinient\" dependencies for efficient parallel builds" ON)

if(X86_64)
    set(ENABLE_INTEL_GPU_DEFAULT ON)
else()
    set(ENABLE_INTEL_GPU_DEFAULT OFF)
endif()

ie_dependent_option (ENABLE_INTEL_GPU "GPU OpenCL-based plugin for OpenVINO Runtime" ${ENABLE_INTEL_GPU_DEFAULT} "X86_64 OR AARCH64;NOT APPLE;NOT MINGW;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)

if (ANDROID OR (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    # oneDNN doesn't support old compilers and android builds for now, so we'll
    # build GPU plugin without oneDNN
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT OFF)
else()
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT ON)
endif()

ie_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ${ENABLE_ONEDNN_FOR_GPU_DEFAULT} "ENABLE_INTEL_GPU" OFF)

ie_option (ENABLE_PROFILING_ITT "Build with ITT tracing. Optionally configure pre-built ittnotify library though INTEL_VTUNE_DIR variable." OFF)

ie_option_enum(ENABLE_PROFILING_FILTER "Enable or disable ITT counter groups.\
Supported values:\
 ALL - enable all ITT counters (default value)\
 FIRST_INFERENCE - enable only first inference time counters" ALL
               ALLOWED_VALUES ALL FIRST_INFERENCE)

ie_option (ENABLE_PROFILING_FIRST_INFERENCE "Build with ITT tracing of first inference time." ON)

ie_option_enum(SELECTIVE_BUILD "Enable OpenVINO conditional compilation or statistics collection. \
In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected InelSEAPI statistics. \
Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv" OFF
               ALLOWED_VALUES ON OFF COLLECT)

ie_option (ENABLE_DOCS "Build docs using Doxygen" OFF)

find_package(PkgConfig QUIET)
ie_dependent_option (ENABLE_PKGCONFIG_GEN "Enable openvino.pc pkg-config file generation" ON "LINUX OR APPLE;PkgConfig_FOUND;BUILD_SHARED_LIBS" OFF)

#
# OpenVINO Runtime specific options
#

# "OneDNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
set(THREADING "TBB" CACHE STRING "Threading")
set_property(CACHE THREADING PROPERTY STRINGS "TBB" "TBB_AUTO" "OMP" "SEQ")
list (APPEND IE_OPTIONS THREADING)
if (NOT THREADING STREQUAL "TBB" AND
    NOT THREADING STREQUAL "TBB_AUTO" AND
    NOT THREADING STREQUAL "OMP" AND
    NOT THREADING STREQUAL "SEQ")
    message(FATAL_ERROR "THREADING should be set to TBB (default), TBB_AUTO, OMP or SEQ")
endif()

if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO") AND
    (BUILD_SHARED_LIBS OR (LINUX AND X86_64)))
    set(ENABLE_TBBBIND_2_5_DEFAULT ON)
else()
    set(ENABLE_TBBBIND_2_5_DEFAULT OFF)
endif()

ie_dependent_option (ENABLE_TBBBIND_2_5 "Enable TBBBind_2_5 static usage in OpenVINO runtime" ${ENABLE_TBBBIND_2_5_DEFAULT} "THREADING MATCHES TBB" OFF)

ie_dependent_option (ENABLE_INTEL_GNA "GNA support for OpenVINO Runtime" ON
    "NOT APPLE;NOT ANDROID;X86_64;CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 5.4" OFF)

ie_option (ENABLE_INTEL_GNA_DEBUG "GNA debug build" OFF)

ie_dependent_option (ENABLE_IR_V7_READER "Enables IR v7 reader" ${BUILD_SHARED_LIBS} "ENABLE_TESTS;ENABLE_INTEL_GNA" OFF)

ie_option (ENABLE_GAPI_PREPROCESSING "Enables G-API preprocessing" ON)

ie_option (ENABLE_MULTI "Enables MULTI Device Plugin" ON)
ie_option (ENABLE_AUTO "Enables AUTO Device Plugin" ON)

ie_option (ENABLE_AUTO_BATCH "Enables Auto-Batching Plugin" ON)

ie_option (ENABLE_HETERO "Enables Hetero Device Plugin" ON)

ie_option (ENABLE_TEMPLATE "Enable template plugin" ON)

ie_dependent_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF "ENABLE_TESTS;ENABLE_GAPI_PREPROCESSING" OFF)

ie_dependent_option (ENABLE_DATA "fetch models from testdata repo" ON "ENABLE_FUNCTIONAL_TESTS;NOT ANDROID" OFF)

ie_dependent_option (ENABLE_BEH_TESTS "tests oriented to check OpenVINO Runtime API correctness" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON "ENABLE_TESTS" OFF)

ie_option (ENABLE_SAMPLES "console samples are part of OpenVINO Runtime package" ON)

ie_option (ENABLE_OPENCV "enables custom OpenCV download" OFF)

ie_option (ENABLE_V7_SERIALIZE "enables serialization to IR v7" OFF)

set(OPENVINO_EXTRA_MODULES "" CACHE STRING "Extra paths for extra modules to include into OpenVINO build")

ie_dependent_option(ENABLE_TBB_RELEASE_ONLY "Only Release TBB libraries are linked to the OpenVINO Runtime binaries" ON "THREADING MATCHES TBB;LINUX" OFF)

if(CMAKE_HOST_LINUX AND LINUX)
    # Debian packages are enabled on Ubuntu systems
    # so, system TBB / pugixml / OpenCL can be tried for usage
    set(ENABLE_SYSTEM_LIBS_DEFAULT ON)
else()
    set(ENABLE_SYSTEM_LIBS_DEFAULT OFF)
endif()

# try to search TBB from brew by default
if(APPLE AND AARCH64)
    set(ENABLE_SYSTEM_TBB_DEFAULT ON)
else()
    set(ENABLE_SYSTEM_TBB_DEFAULT ${ENABLE_SYSTEM_LIBS_DEFAULT})
endif()

# users wants to use his own TBB version, specific either via env vars or cmake options
if(DEFINED ENV{TBBROOT} OR DEFINED ENV{TBB_DIR} OR DEFINED TBB_DIR OR DEFINED TBBROOT)
    set(ENABLE_SYSTEM_TBB_DEFAULT OFF)
endif()

# for static libraries case libpugixml.a must be compiled with -fPIC
ie_dependent_option (ENABLE_SYSTEM_PUGIXML "use the system copy of pugixml" ${ENABLE_SYSTEM_LIBS_DEFAULT} "BUILD_SHARED_LIBS" OFF)

ie_dependent_option (ENABLE_SYSTEM_TBB  "use the system version of TBB" ${ENABLE_SYSTEM_TBB_DEFAULT} "THREADING MATCHES TBB" OFF)

ie_dependent_option (ENABLE_SYSTEM_OPENCL "Use the system version of OpenCL" ${ENABLE_SYSTEM_LIBS_DEFAULT} "BUILD_SHARED_LIBS;ENABLE_INTEL_GPU" OFF)

ie_option (ENABLE_DEBUG_CAPS "enable OpenVINO debug capabilities at runtime" OFF)

ie_dependent_option (ENABLE_GPU_DEBUG_CAPS "enable GPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS" OFF)

ie_dependent_option (ENABLE_CPU_DEBUG_CAPS "enable CPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS" OFF)

find_host_package(PythonInterp 3 QUIET)
ie_option(ENABLE_OV_ONNX_FRONTEND "Enable ONNX FrontEnd" ${PYTHONINTERP_FOUND})
ie_option(ENABLE_OV_PADDLE_FRONTEND "Enable PaddlePaddle FrontEnd" ON)
ie_option(ENABLE_OV_IR_FRONTEND "Enable IR FrontEnd" ON)
ie_option(ENABLE_OV_PYTORCH_FRONTEND "Enable PyTorch FrontEnd" ON)
ie_option(ENABLE_OV_TF_FRONTEND "Enable TensorFlow FrontEnd" ON)
ie_option(ENABLE_OV_TF_LITE_FRONTEND "Enable TensorFlow Lite FrontEnd" ON)
ie_dependent_option(ENABLE_SYSTEM_PROTOBUF "Use system protobuf" OFF
    "ENABLE_OV_ONNX_FRONTEND OR ENABLE_OV_PADDLE_FRONTEND OR ENABLE_OV_TF_FRONTEND;BUILD_SHARED_LIBS" OFF)
ie_option(ENABLE_OV_IR_FRONTEND "Enable IR FrontEnd" ON)
ie_dependent_option(ENABLE_SYSTEM_FLATBUFFERS "Use system flatbuffers" ON
    "ENABLE_OV_TF_LITE_FRONTEND" OFF)

ie_dependent_option(ENABLE_OV_CORE_UNIT_TESTS "Enables OpenVINO core unit tests" ON "ENABLE_TESTS" OFF)
ie_option(ENABLE_OPENVINO_DEBUG "Enable output for OPENVINO_DEBUG statements" OFF)

if(NOT BUILD_SHARED_LIBS AND ENABLE_OV_TF_FRONTEND)
    set(FORCE_FRONTENDS_USE_PROTOBUF ON)
else()
    set(FORCE_FRONTENDS_USE_PROTOBUF OFF)
endif()

#
# Process featues
#

if(ENABLE_OPENVINO_DEBUG)
    add_definitions(-DENABLE_OPENVINO_DEBUG)
endif()

if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

print_enabled_features()
