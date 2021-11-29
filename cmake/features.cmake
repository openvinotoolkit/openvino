# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
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
# enable or disable output from NGRAPH_DEBUG statements
#

if(NGRAPH_DEBUG_ENABLE)
    add_definitions(-DNGRAPH_DEBUG_ENABLE)
endif()

#
# Process options
#

print_enabled_features()
