# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include (target_flags)
include (options)

#these options are aimed to optimize build time on development system
ie_option (ENABLE_NGRAPH "Enable nGraph build" ON)
ie_option (ENABLE_INFERENCE_ENGINE "Enable Inference Engine build" ON)
ie_option (ENABLE_DOCKER "docker images" OFF)
ie_option (ENABLE_LTO "Enable Link Time Optimization" OFF)
