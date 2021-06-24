# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Common cmake options
#

ie_dependent_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON "X86_64" OFF)

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_option (ENABLE_STRICT_DEPENDENCIES "Skip configuring \"convinient\" dependencies for efficient parallel builds" ON)

ie_dependent_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON "X86_64;NOT APPLE;NOT MINGW;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)

ie_option (ENABLE_PROFILING_ITT "Build with ITT tracing. Optionally configure pre-built ittnotify library though INTEL_VTUNE_DIR variable." OFF)

ie_option_enum(ENABLE_PROFILING_FILTER "Enable or disable ITT counter groups.\
Supported values:\
 ALL - enable all ITT counters (default value)\
 FIRST_INFERENCE - enable only first inference time counters" ALL
               ALLOWED_VALUES ALL FIRST_INFERENCE)

ie_option (ENABLE_PROFILING_FIRST_INFERENCE "Build with ITT tracing of first inference time." ON)

ie_option(ENABLE_TEMPLATE_PLUGIN "Register template plugin into plugins.xml" OFF)

ie_option_enum(SELECTIVE_BUILD "Enable OpenVINO conditional compilation or statistics collection. \
In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected InelSEAPI statistics. \
Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv" OFF
               ALLOWED_VALUES ON OFF COLLECT)

ie_option(ENABLE_ERROR_HIGHLIGHT "Highlight errors and warnings during compile time" OFF)

# Try to find python3
find_package(PythonLibs 3 QUIET)
ie_dependent_option (ENABLE_PYTHON "enables ie python bridge build" OFF "PYTHONLIBS_FOUND" OFF)

find_package(PythonInterp 3 QUIET)
ie_dependent_option (ENABLE_DOCS "Build docs using Doxygen" OFF "PYTHONINTERP_FOUND" OFF)

#
# Inference Engine specific options
#

ie_dependent_option (ENABLE_GNA "GNA support for inference engine" ON "NOT APPLE;NOT ANDROID;X86_64" OFF)

ie_dependent_option (ENABLE_CLDNN_TESTS "Enable clDNN unit tests" OFF "ENABLE_CLDNN" OFF)

# "MKL-DNN library based on OMP or TBB or Sequential implementation: TBB|OMP|SEQ"
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

if (ENABLE_GNA)
    if (UNIX AND NOT APPLE AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        set (DEFAULT_GNA_LIB GNA1)
    else()
        set (DEFAULT_GNA_LIB GNA2)
    endif()
    set(GNA_LIBRARY_VERSION "${DEFAULT_GNA_LIB}" CACHE STRING "GNAVersion")
    set_property(CACHE GNA_LIBRARY_VERSION PROPERTY STRINGS "GNA1" "GNA1_1401" "GNA2")
    list (APPEND IE_OPTIONS GNA_LIBRARY_VERSION)
    if (NOT GNA_LIBRARY_VERSION STREQUAL "GNA1" AND
        NOT GNA_LIBRARY_VERSION STREQUAL "GNA1_1401" AND
        NOT GNA_LIBRARY_VERSION STREQUAL "GNA2")
        message(FATAL_ERROR "GNA_LIBRARY_VERSION should be set to GNA1, GNA1_1401 or GNA2. Default option is ${DEFAULT_GNA_LIB}")
    endif()
endif()

ie_dependent_option (ENABLE_VPU "vpu targeted plugins for inference engine" ON "NOT WINDOWS_PHONE;NOT WINDOWS_STORE" OFF)

ie_dependent_option (ENABLE_MYRIAD "myriad targeted plugin for inference engine" ON "ENABLE_VPU" OFF)

ie_dependent_option (ENABLE_MYRIAD_NO_BOOT "myriad plugin will skip device boot" OFF "ENABLE_MYRIAD" OFF)

ie_dependent_option (ENABLE_GAPI_TESTS "tests for GAPI kernels" ON "ENABLE_TESTS" OFF)

ie_dependent_option (GAPI_TEST_PERF "if GAPI unit tests should examine performance" OFF "ENABLE_GAPI_TESTS" OFF)

ie_dependent_option (ENABLE_MYRIAD_MVNC_TESTS "functional and behavior tests for mvnc api" OFF "ENABLE_TESTS;ENABLE_MYRIAD" OFF)

ie_dependent_option (ENABLE_DATA "fetch models from testdata repo" ON "ENABLE_FUNCTIONAL_TESTS;NOT ANDROID" OFF)

ie_dependent_option (ENABLE_BEH_TESTS "tests oriented to check inference engine API corecteness" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_FUNCTIONAL_TESTS "functional tests" ON "ENABLE_TESTS" OFF)

ie_dependent_option (ENABLE_SAMPLES "console samples are part of inference engine package" ON "NOT MINGW" OFF)

ie_dependent_option (ENABLE_SPEECH_DEMO "enable speech demo integration" ON "NOT APPLE;NOT ANDROID;X86 OR X86_64" OFF)

ie_option (ENABLE_OPENCV "enables OpenCV" ON)

ie_option (ENABLE_V7_SERIALIZE "enables serialization to IR v7" OFF)

set(IE_EXTRA_MODULES "" CACHE STRING "Extra paths for extra modules to include into OpenVINO build")

ie_dependent_option(ENABLE_TBB_RELEASE_ONLY "Only Release TBB libraries are linked to the Inference Engine binaries" ON "THREADING MATCHES TBB;LINUX" OFF)

ie_option (USE_SYSTEM_PUGIXML "use the system copy of pugixml" OFF)

ie_option (ENABLE_CPU_DEBUG_CAPS "enable CPU debug capabilities at runtime" OFF)

#
# Process featues
#

if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

if (ENABLE_PROFILING_RAW)
    add_definitions(-DENABLE_PROFILING_RAW=1)
endif()

if (ENABLE_MYRIAD)
    add_definitions(-DENABLE_MYRIAD=1)
endif()

if (ENABLE_MYRIAD_NO_BOOT AND ENABLE_MYRIAD )
    add_definitions(-DENABLE_MYRIAD_NO_BOOT=1)
endif()

if (ENABLE_CLDNN)
    add_definitions(-DENABLE_CLDNN=1)
endif()

if (ENABLE_MKL_DNN)
    add_definitions(-DENABLE_MKL_DNN=1)
endif()

if (ENABLE_GNA)
    add_definitions(-DENABLE_GNA)

    if (UNIX AND NOT APPLE AND CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        message(WARNING "${GNA_LIBRARY_VERSION} is not supported on GCC version ${CMAKE_CXX_COMPILER_VERSION}. Fallback to GNA1")
        set(GNA_LIBRARY_VERSION GNA1)
    endif()
endif()

print_enabled_features()
