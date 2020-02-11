# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (target_flags)
include (options)

# these options are aimed to optimize build time on development system

ie_option (ENABLE_NGRAPH "Enable nGraph build" ON)

ie_option (ENABLE_INFERENCE_ENGINE "Enable Inference Engine build" ON)

ie_option (ENABLE_MKL_DNN "MKL-DNN plugin for inference engine" ON)

ie_option (ENABLE_CLDNN "clDnn based plugin for inference engine" ON)

ie_option (ENABLE_LTO "Enable Link Time Optimization" OFF)

ie_option (OS_FOLDER "create OS dedicated folder in output" OFF)

ie_option (TREAT_WARNING_AS_ERROR "Treat build warnings as errors" OFF)

ie_option (ENABLE_SANITIZER "enable checking memory errors via AddressSanitizer" OFF)

ie_option (ENABLE_THREAD_SANITIZER "enable checking data races via ThreadSanitizer" OFF)
