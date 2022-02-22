// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifdef _WIN32
#        if defined __INTEL_COMPILER || defined _MSC_VER
#            define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        endif
#    elif defined(__GNUC__) && (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ > 2)) || defined(__clang__)
#        define OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif
