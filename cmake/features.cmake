# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common cmake options
#

ov_option (ENABLE_PROXY "Proxy plugin for OpenVINO Runtime" ON)

if(WIN32 AND AARCH64 AND OV_COMPILER_IS_CLANG)
    set(ENABLE_INTEL_CPU_DEFAULT OFF)
else()
    set(ENABLE_INTEL_CPU_DEFAULT ON)
endif()

ov_dependent_option (ENABLE_INTEL_CPU "CPU plugin for OpenVINO Runtime" ${ENABLE_INTEL_CPU_DEFAULT}
    "RISCV64 OR X86 OR X86_64 OR AARCH64 OR ARM" OFF)

ov_dependent_option (ENABLE_ARM_COMPUTE_CMAKE "Enable ARM Compute build via cmake" OFF "ENABLE_INTEL_CPU" OFF)

ov_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

if(ENABLE_TESTS)
    include(CTest)
    enable_testing()
endif()

if(X86_64)
    set(ENABLE_INTEL_GPU_DEFAULT ON)
else()
    set(ENABLE_INTEL_GPU_DEFAULT OFF)
endif()

ov_dependent_option (ENABLE_INTEL_GPU "GPU OpenCL-based plugin for OpenVINO Runtime" ${ENABLE_INTEL_GPU_DEFAULT} "X86_64 OR AARCH64;NOT APPLE;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)

if (ANDROID OR MINGW OR (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0) OR (NOT BUILD_SHARED_LIBS AND ENABLE_INTEL_CPU))
    # oneDNN doesn't support old compilers and android builds for now, so we'll build GPU plugin without oneDNN
    # also, in case of static build CPU's and GPU's oneDNNs will conflict, so we are disabling GPU's one in this case
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT OFF)
else()
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT ON)
endif()

ov_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ${ENABLE_ONEDNN_FOR_GPU_DEFAULT} "ENABLE_INTEL_GPU" OFF)

ov_option (ENABLE_DEBUG_CAPS "enable OpenVINO debug capabilities at runtime" OFF)
ov_dependent_option (ENABLE_GPU_DEBUG_CAPS "enable GPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS;ENABLE_INTEL_GPU" OFF)
ov_dependent_option (ENABLE_CPU_DEBUG_CAPS "enable CPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS;ENABLE_INTEL_CPU" OFF)

ov_option (ENABLE_PROFILING_ITT "Build with ITT tracing. Optionally configure pre-built ittnotify library though INTEL_VTUNE_DIR variable." OFF)

ov_option_enum(ENABLE_PROFILING_FILTER "Enable or disable ITT counter groups.\
Supported values:\
 ALL - enable all ITT counters (default value)\
 FIRST_INFERENCE - enable only first inference time counters" ALL
               ALLOWED_VALUES ALL FIRST_INFERENCE)

ov_option (ENABLE_PROFILING_FIRST_INFERENCE "Build with ITT tracing of first inference time." ON)

ov_option_enum(SELECTIVE_BUILD "Enable OpenVINO conditional compilation or statistics collection. \
In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected IntelSEAPI statistics. \
Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv" OFF
               ALLOWED_VALUES ON OFF COLLECT)

ov_option (ENABLE_DOCS "Build docs using Doxygen" OFF)

find_package(PkgConfig QUIET)
ov_dependent_option (ENABLE_PKGCONFIG_GEN "Enable openvino.pc pkg-config file generation" ON "LINUX OR APPLE;PkgConfig_FOUND;BUILD_SHARED_LIBS" OFF)

#
# OpenVINO Runtime specific options
#

# "OneDNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if(RISCV64)
    # oneDNN does not support non-SEQ for RISC-V architecture
    set(THREADING_DEFAULT "SEQ")
else()
    set(THREADING_DEFAULT "TBB")
endif()

set(THREADING_OPTIONS "TBB" "TBB_AUTO" "SEQ")
if(NOT APPLE)
    list(APPEND THREADING_OPTIONS "OMP")
endif()

set(THREADING "${THREADING_DEFAULT}" CACHE STRING "Threading")
set_property(CACHE THREADING PROPERTY STRINGS ${THREADING_OPTIONS})
list (APPEND OV_OPTIONS THREADING)
if(NOT THREADING IN_LIST THREADING_OPTIONS)
    message(FATAL_ERROR "THREADING should be set to either ${THREADING_OPTIONS}")
endif()

if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO") AND
    (BUILD_SHARED_LIBS OR (LINUX AND X86_64)))
    set(ENABLE_TBBBIND_2_5_DEFAULT ON)
else()
    set(ENABLE_TBBBIND_2_5_DEFAULT OFF)
endif()

ov_dependent_option (ENABLE_TBBBIND_2_5 "Enable TBBBind_2_5 static usage in OpenVINO runtime" ${ENABLE_TBBBIND_2_5_DEFAULT} "THREADING MATCHES TBB; NOT APPLE" OFF)
ov_dependent_option (ENABLE_TBB_RELEASE_ONLY "Only Release TBB libraries are linked to the OpenVINO Runtime binaries" ON "THREADING MATCHES TBB;LINUX" OFF)

ov_dependent_option (ENABLE_INTEL_GNA "GNA support for OpenVINO Runtime" ON
    "NOT APPLE;NOT ANDROID;X86_64;CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 5.4" OFF)

ov_dependent_option (ENABLE_INTEL_GNA_DEBUG "GNA debug build" OFF "ENABLE_INTEL_GNA" OFF)
ov_dependent_option (ENABLE_V7_SERIALIZE "enables serialization to IR v7" OFF "ENABLE_INTEL_GNA" OFF)
ov_dependent_option (ENABLE_IR_V7_READER "Enables IR v7 reader" ${BUILD_SHARED_LIBS} "ENABLE_TESTS;ENABLE_INTEL_GNA" OFF)

ov_dependent_option (ENABLE_GAPI_PREPROCESSING "Enables G-API preprocessing" ON "NOT MINGW64" OFF)

ov_option (ENABLE_MULTI "Enables MULTI Device Plugin" ON)
ov_option (ENABLE_AUTO "Enables AUTO Device Plugin" ON)
ov_option (ENABLE_AUTO_BATCH "Enables Auto-Batching Plugin" ON)
ov_option (ENABLE_HETERO "Enables Hetero Device Plugin" ON)
ov_option (ENABLE_TEMPLATE "Enable template plugin" ON)

ov_dependent_option (ENABLE_PLUGINS_XML "Generate plugins.xml configuration file or not" OFF "BUILD_SHARED_LIBS" OFF)

ov_dependent_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF "ENABLE_TESTS;ENABLE_GAPI_PREPROCESSING" OFF)

ov_dependent_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON "ENABLE_TESTS" OFF)

ov_option (ENABLE_SAMPLES "console samples are part of OpenVINO Runtime package" ON)

set(OPENVINO_EXTRA_MODULES "" CACHE STRING "Extra paths for extra modules to include into OpenVINO build")

find_host_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    ov_option(ENABLE_OV_ONNX_FRONTEND "Enable ONNX FrontEnd" ON)
else()
    ov_option(ENABLE_OV_ONNX_FRONTEND "Enable ONNX FrontEnd" OFF)
endif()
ov_option(ENABLE_OV_PADDLE_FRONTEND "Enable PaddlePaddle FrontEnd" ON)
ov_option(ENABLE_OV_IR_FRONTEND "Enable IR FrontEnd" ON)
ov_option(ENABLE_OV_PYTORCH_FRONTEND "Enable PyTorch FrontEnd" ON)
ov_option(ENABLE_OV_IR_FRONTEND "Enable IR FrontEnd" ON)
ov_option(ENABLE_OV_TF_FRONTEND "Enable TensorFlow FrontEnd" ON)
ov_option(ENABLE_OV_TF_LITE_FRONTEND "Enable TensorFlow Lite FrontEnd" ON)
ov_dependent_option(ENABLE_SNAPPY_COMPRESSION "Enables compression support for TF FE" ON
    "ENABLE_OV_TF_FRONTEND" OFF)

ov_dependent_option (ENABLE_STRICT_DEPENDENCIES "Skip configuring \"convinient\" dependencies for efficient parallel builds" ON "ENABLE_TESTS;ENABLE_OV_ONNX_FRONTEND" OFF)

if(CMAKE_HOST_LINUX AND LINUX)
    # Debian packages are enabled on Ubuntu systems
    # so, system TBB / pugixml / OpenCL can be tried for usage
    set(ENABLE_SYSTEM_LIBS_DEFAULT ON)
else()
    set(ENABLE_SYSTEM_LIBS_DEFAULT OFF)
endif()

if(ANDROID)
    # when protobuf from /usr/include is used, then Android toolchain ignores include paths
    # but if we build for Android using vcpkg / conan / etc where flatbuffers is not located in
    # the /usr/include folders, we can still use 'system' flatbuffers
    set(ENABLE_SYSTEM_FLATBUFFERS_DEFAULT OFF)
else()
    set(ENABLE_SYSTEM_FLATBUFFERS_DEFAULT ON)
endif()

# users wants to use his own TBB version, specific either via env vars or cmake options
if(DEFINED ENV{TBBROOT} OR DEFINED ENV{TBB_DIR} OR DEFINED TBB_DIR OR DEFINED TBBROOT)
    set(ENABLE_SYSTEM_TBB_DEFAULT OFF)
else()
    set(ENABLE_SYSTEM_TBB_DEFAULT ${ENABLE_SYSTEM_LIBS_DEFAULT})
endif()

ov_dependent_option (ENABLE_SYSTEM_TBB  "Enables use of system TBB" ${ENABLE_SYSTEM_TBB_DEFAULT}
    "THREADING MATCHES TBB" OFF)
ov_option (ENABLE_SYSTEM_PUGIXML "Enables use of system PugiXML" OFF)
# the option is on by default, because we use only flatc compiler and don't use any libraries
ov_dependent_option(ENABLE_SYSTEM_FLATBUFFERS "Enables use of system flatbuffers" ${ENABLE_SYSTEM_FLATBUFFERS_DEFAULT}
    "ENABLE_OV_TF_LITE_FRONTEND" OFF)
ov_dependent_option (ENABLE_SYSTEM_OPENCL "Enables use of system OpenCL" ${ENABLE_SYSTEM_LIBS_DEFAULT}
    "ENABLE_INTEL_GPU" OFF)
# the option is turned off by default, because we compile our own static version of protobuf
# with LTO and -fPIC options, while system one does not have such flags
ov_dependent_option (ENABLE_SYSTEM_PROTOBUF "Enables use of system Protobuf" OFF
    "ENABLE_OV_ONNX_FRONTEND OR ENABLE_OV_PADDLE_FRONTEND OR ENABLE_OV_TF_FRONTEND" OFF)
# the option is turned off by default, because we don't want to have a dependency on libsnappy.so
ov_dependent_option (ENABLE_SYSTEM_SNAPPY "Enables use of system version of Snappy" OFF
    "ENABLE_SNAPPY_COMPRESSION" OFF)

ov_dependent_option (ENABLE_PYTHON_PACKAGING "Enables packaging of Python API in APT / YUM" OFF
    "ENABLE_PYTHON;UNIX" OFF)

ov_option(ENABLE_OPENVINO_DEBUG "Enable output for OPENVINO_DEBUG statements" OFF)

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

ov_print_enabled_features()
