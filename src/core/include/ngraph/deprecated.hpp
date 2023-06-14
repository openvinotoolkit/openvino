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

#include "openvino/core/deprecated.hpp"

#define NGRAPH_DEPRECATED(msg)           OPENVINO_DEPRECATED(msg)
#define NGRAPH_ENUM_DEPRECATED(msg)      OPENVINO_ENUM_DEPRECATED(msg)
#define NGRAPH_SUPPRESS_DEPRECATED_START OPENVINO_SUPPRESS_DEPRECATED_START
#define NGRAPH_SUPPRESS_DEPRECATED_END   OPENVINO_SUPPRESS_DEPRECATED_END
#define NGRAPH_API_DEPRECATED                                                                      \
    OPENVINO_DEPRECATED("The nGraph API is deprecated and will be removed in the 2024.0 release. " \
                        "For instructions on transitioning to the new API, please refer to "       \
                        "https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
