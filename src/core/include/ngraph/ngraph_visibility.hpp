// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include "ngraph/visibility.hpp"
#include "openvino/core/core_visibility.hpp"

#define NGRAPH_API      OPENVINO_API
#define NGRAPH_API_C    OPENVINO_API_C
#define NGRAPH_EXTERN_C OPENVINO_EXTERN_C

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#    ifndef ENABLE_UNICODE_PATH_SUPPORT
#        define ENABLE_UNICODE_PATH_SUPPORT
#    endif
#endif
