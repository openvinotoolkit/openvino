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

#include <memory>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ngraph {
namespace op {
using ov::op::util::is_binary_elementwise_arithmetic;
using ov::op::util::is_binary_elementwise_comparison;
using ov::op::util::is_binary_elementwise_logical;
using ov::op::util::is_commutative;
using ov::op::util::is_constant;
using ov::op::util::is_op;
using ov::op::util::is_output;
using ov::op::util::is_parameter;
using ov::op::util::is_sink;
using ov::op::util::is_unary_elementwise_arithmetic;
using ov::op::util::supports_auto_broadcast;
}  // namespace op
}  // namespace ngraph
