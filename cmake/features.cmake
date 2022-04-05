# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common cmake options
#

ie_dependent_option (ENABLE_INTEL_CPU "CPU plugin for inference engine" ON "X86_64" OFF)

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_option (ENABLE_STRICT_DEPENDENCIES "Skip configuring \"convinient\" dependencies for efficient parallel builds" ON)

ie_dependent_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON "X86_64;NOT APPLE;NOT MINGW;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)
ie_dependent_option (ENABLE_INTEL_GPU "GPU plugin for inference engine on Intel GPU" ON "ENABLE_CLDNN" OFF)

if (NOT ENABLE_CLDNN OR ANDROID OR
    (CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    # oneDNN doesn't support old compilers and android builds for now, so we'll
    # build GPU plugin without oneDNN
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT OFF)
else()
    set(ENABLE_ONEDNN_FOR_GPU_DEFAULT ON)
endif()

ie_dependent_option (ENABLE_ONEDNN_FOR_GPU "Enable oneDNN with GPU support" ON "ENABLE_ONEDNN_FOR_GPU_DEFAULT" OFF)

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

ie_option(ENABLE_ERROR_HIGHLIGHT "Highlight errors and warnings during compile time" OFF)

# Try to find python3
find_package(PythonLibs 3 QUIET)
# Check for Cython to build IE_API 
ov_check_pip_packages(REQUIREMENTS_FILE ${CMAKE_SOURCE_DIR}/src/bindings/python/requirements.txt
                     RESULT_VAR OV_PYTHON_REQS)
ov_check_pip_package(REQUIREMENT "cython>=0.29.22"
                     RESULT_VAR CYTHON)

ie_dependent_option (ENABLE_PYTHON "enables ie python bridge build" ON "PYTHONLIBS_FOUND;CYTHON;OV_PYTHON_REQS" OFF)

find_package(PythonInterp 3 QUIET)
ie_dependent_option (ENABLE_DOCS "Build docs using Doxygen" OFF "PYTHONINTERP_FOUND" OFF)

# Check for wheel package
ov_check_pip_packages(REQUIREMENTS_FILE ${CMAKE_SOURCE_DIR}/src/bindings/python/wheel/requirements-dev.txt
                     RESULT_VAR WHEEL_REQS
                     FAIL_FAST)
                     
set (WHEEL_CONDITION "PYTHONINTERP_FOUND;ENABLE_PYTHON;WHEEL_REQS;CMAKE_SOURCE_DIR STREQUAL OpenVINO_SOURCE_DIR")

if(LINUX)
    find_host_program(patchelf_program
                      NAMES patchelf
                      DOC "Path to patchelf tool")
    if(NOT patchelf_program)
        message(WARNING "patchelf is not found. It is required to build OpenVINO Runtime wheel")
        list(APPEND WHEEL_CONDITION patchelf_program)
    endif()
endif()

# this option should not be a part of InferenceEngineDeveloperPackage
# since wheels can be built only together with main OV build
ie_dependent_option (ENABLE_WHEEL "Build wheel packages for PyPI" ON "${WHEEL_CONDITION}" OFF)

#
# Inference Engine specific options
#

# "OneDNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
if(X86 OR ARM OR (MSVC AND (ARM OR AARCH64)) )
    set(THREADING_DEFAULT "SEQ")
else()
    set(THREADING_DEFAULT "TBB")
endif()
set(THREADING "${THREADING_DEFAULT}" CACHE STRING "Threading")
set_property(CACHE THREADING PROPERTY STRINGS "TBB" "TBB_AUTO" "OMP" "SEQ")
list (APPEND IE_OPTIONS THREADING)
if (NOT THREADING STREQUAL "TBB" AND
    NOT THREADING STREQUAL "TBB_AUTO" AND
    NOT THREADING STREQUAL "OMP" AND
    NOT THREADING STREQUAL "SEQ")
    message(FATAL_ERROR "THREADING should be set to TBB, TBB_AUTO, OMP or SEQ. Default option is ${THREADING_DEFAULT}")
endif()

if((THREADING STREQUAL "TBB" OR THREADING STREQUAL "TBB_AUTO") AND
    (BUILD_SHARED_LIBS OR (LINUX AND X86_64)))
    set(ENABLE_TBBBIND_2_5_DEFAULT ON)
else()
    set(ENABLE_TBBBIND_2_5_DEFAULT OFF)
endif()

ie_dependent_option (ENABLE_TBBBIND_2_5 "Enable TBBBind_2_5 static usage in OpenVINO runtime" ${ENABLE_TBBBIND_2_5_DEFAULT} "THREADING MATCHES TBB" OFF)

ie_dependent_option (ENABLE_INTEL_GNA "GNA support for inference engine" ON
    "NOT APPLE;NOT ANDROID;X86_64;CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 5.4" OFF)

if(ENABLE_TESTS OR BUILD_SHARED_LIBS)
    set(ENABLE_IR_V7_READER_DEFAULT ON)
else()
    set(ENABLE_IR_V7_READER_DEFAULT OFF)
endif()

ie_option (ENABLE_IR_V7_READER "Enables IR v7 reader" ${ENABLE_IR_V7_READER_DEFAULT})

ie_option (ENABLE_GAPI_PREPROCESSING "Enables G-API preprocessing" ON)

ie_option (ENABLE_MULTI "Enables MULTI Device Plugin" ON)
ie_option (ENABLE_AUTO "Enables AUTO Device Plugin" ON)

ie_option (ENABLE_AUTO_BATCH "Enables Auto-Batching Plugin" ON)

ie_option (ENABLE_HETERO "Enables Hetero Device Plugin" ON)

ie_option (ENABLE_TEMPLATE "Enable template plugin" ON)

ie_dependent_option (ENABLE_INTEL_MYRIAD_COMMON "common part of myriad plugin" ON "NOT WINDOWS_PHONE;NOT WINDOWS_STORE" OFF)

ie_dependent_option (ENABLE_INTEL_MYRIAD "myriad targeted plugin for inference engine" ON "ENABLE_INTEL_MYRIAD_COMMON" OFF)

ie_dependent_option (ENABLE_MYRIAD_NO_BOOT "myriad plugin will skip device boot" OFF "ENABLE_INTEL_MYRIAD" OFF)

ie_dependent_option (ENABLE_GAPI_TESTS "tests for GAPI kernels" ON "ENABLE_GAPI_PREPROCESSING;ENABLE_TESTS" OFF)

ie_dependent_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF "ENABLE_GAPI_TESTS" OFF)

ie_dependent_option (ENABLE_MYRIAD_MVNC_TESTS "functional and behavior tests for mvnc api" OFF "ENABLE_TESTS;ENABLE_INTEL_MYRIAD" OFF)

ie_dependent_option (ENABLE_DATA "fetch models from testdata repo" ON "ENABLE_FUNCTIONAL_TESTS;NOT ANDROID" OFF)

ie_dependent_option (ENABLE_BEH_TESTS "tests oriented to check inference engine API corecteness" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON "NOT MINGW" OFF)

ie_option (ENABLE_OPENCV "enables OpenCV" ON)

ie_option (ENABLE_V7_SERIALIZE "enables serialization to IR v7" OFF)

set(IE_EXTRA_MODULES "" CACHE STRING "Extra paths for extra modules to include into OpenVINO build")

ie_dependent_option(ENABLE_TBB_RELEASE_ONLY "Only Release TBB libraries are linked to the Inference Engine binaries" ON "THREADING MATCHES TBB;LINUX" OFF)

ie_dependent_option (ENABLE_SYSTEM_PUGIXML "use the system copy of pugixml" OFF "BUILD_SHARED_LIBS" OFF)

get_linux_name(LINUX_OS_NAME)
if(LINUX_OS_NAME MATCHES "^Ubuntu [0-9]+\.[0-9]+$" AND NOT DEFINED ENV{TBBROOT})
    # Debian packages are enabled on Ubuntu systems
    # so, system TBB can be tried for usage
    set(ENABLE_SYSTEM_TBB_DEFAULT ON)
else()
    set(ENABLE_SYSTEM_TBB_DEFAULT OFF)
endif()

ie_dependent_option (ENABLE_SYSTEM_TBB  "use the system version of TBB" ${ENABLE_SYSTEM_TBB_DEFAULT} "THREADING MATCHES TBB;LINUX" OFF)

ie_option (ENABLE_DEBUG_CAPS "enable OpenVINO debug capabilities at runtime" OFF)

ie_dependent_option (ENABLE_GPU_DEBUG_CAPS "enable GPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS" OFF)

ie_dependent_option (ENABLE_CPU_DEBUG_CAPS "enable CPU debug capabilities at runtime" ON "ENABLE_DEBUG_CAPS" OFF)

if(ANDROID OR WINDOWS_STORE OR (MSVC AND (ARM OR AARCH64)))
    set(protoc_available OFF)
else()
    set(protoc_available ON)
endif()

ie_dependent_option(ENABLE_OV_ONNX_FRONTEND "Enable ONNX FrontEnd" ON "protoc_available" OFF)
ie_dependent_option(ENABLE_OV_PADDLE_FRONTEND "Enable PaddlePaddle FrontEnd" ON "protoc_available" OFF)
ie_option(ENABLE_OV_IR_FRONTEND "Enable IR FrontEnd" ON)
ie_dependent_option(ENABLE_OV_TF_FRONTEND "Enable TensorFlow FrontEnd" ON "protoc_available" OFF)
ie_dependent_option(ENABLE_SYSTEM_PROTOBUF "Use system protobuf" OFF
    "ENABLE_OV_ONNX_FRONTEND OR ENABLE_OV_PADDLE_FRONTEND OR ENABLE_OV_TF_FRONTEND;BUILD_SHARED_LIBS" OFF)
ie_dependent_option(ENABLE_OV_CORE_UNIT_TESTS "Enables OpenVINO core unit tests" ON "ENABLE_TESTS;NOT ANDROID" OFF)
ie_dependent_option(ENABLE_OV_CORE_BACKEND_UNIT_TESTS "Control the building of unit tests using backends" ON
    "ENABLE_OV_CORE_UNIT_TESTS" OFF)
ie_option(ENABLE_OPENVINO_DEBUG "Enable output for OPENVINO_DEBUG statements" OFF)
ie_option(ENABLE_REQUIREMENTS_INSTALL "Dynamic dependencies install" ON)

if(NOT BUILD_SHARED_LIBS AND ENABLE_OV_TF_FRONTEND)
    set(FORCE_FRONTENDS_USE_PROTOBUF ON)
else()
    set(FORCE_FRONTENDS_USE_PROTOBUF OFF)
endif()

# WA for ngraph python build on Windows debug
list(REMOVE_ITEM IE_OPTIONS ENABLE_OV_CORE_UNIT_TESTS ENABLE_OV_CORE_BACKEND_UNIT_TESTS)

#
# Process featues
#

if(ENABLE_OPENVINO_DEBUG)
    add_definitions(-DENABLE_OPENVINO_DEBUG)
endif()

if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_INTEL_MYRIAD)
    add_definitions(-DENABLE_INTEL_MYRIAD=1)
endif()

if (ENABLE_MYRIAD_NO_BOOT AND ENABLE_INTEL_MYRIAD)
    add_definitions(-DENABLE_MYRIAD_NO_BOOT=1)
endif()

if (ENABLE_INTEL_GPU)
    add_definitions(-DENABLE_INTEL_GPU=1)
endif()

if (ENABLE_INTEL_CPU)
    add_definitions(-DENABLE_INTEL_CPU=1)
endif()

if (ENABLE_INTEL_GNA)
    add_definitions(-DENABLE_INTEL_GNA)
endif()

print_enabled_features()
