# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(X86_64)
    set(ENABLE_MKL_DNN_DEFAULT ON)
else()
    set(ENABLE_MKL_DNN_DEFAULT OFF)
endif()

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ${ENABLE_MKL_DNN_DEFAULT})

ie_option (ENABLE_TESTS "unit, behavior and functional tests" OFF)

ie_dependent_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON "X86_64;NOT APPLE;NOT MINGW;NOT WINDOWS_STORE;NOT WINDOWS_PHONE" OFF)

ie_option (ENABLE_PROFILING_ITT "Build with ITT tracing. Optionally configure pre-built ittnotify library though INTEL_VTUNE_DIR variable." OFF)

ie_option (ENABLE_DOCS "Build docs using Doxygen" OFF)

ie_option(ENABLE_TEMPLATE_PLUGIN "Register template plugin into plugins.xml" OFF)

ie_option_enum(SELECTIVE_BUILD "Enable OpenVINO conditional compilation or statistics collection. \
In case SELECTIVE_BUILD is enabled, the SELECTIVE_BUILD_STAT variable should contain the path to the collected InelSEAPI statistics. \
Usage: -DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv" OFF
               ALLOWED_VALUES ON OFF COLLECT)

#
# Process options
#

print_enabled_features()
