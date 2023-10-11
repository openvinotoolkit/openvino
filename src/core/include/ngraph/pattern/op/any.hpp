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

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/any.hpp"

namespace ngraph {
namespace pattern {
namespace op {
using ov::pass::pattern::op::Any;
}  // namespace op
}  // namespace pattern
}  // namespace ngraph
