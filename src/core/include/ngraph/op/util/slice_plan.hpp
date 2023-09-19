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

// vpux-plugin doesn't link to openvino::core::dev, thus '../dev_api/' is needed
#include "../dev_api/openvino/op/util/slice_plan.hpp"
#include "ngraph/deprecated.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START
namespace ngraph {
using SlicePlan = ov::op::util::SlicePlan;
using ov::op::util::make_slice_plan;
}  // namespace ngraph

NGRAPH_SUPPRESS_DEPRECATED_END
