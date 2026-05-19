// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/pass/node_registry.hpp"

namespace ov {
namespace decompositions {
namespace detail {

// Create a scalar constant typed as `x` so that pattern matchers in the
// corresponding fusion transformations (which expect a raw `Constant` or
// `Convert(Constant)`, not `ConvertLike`) can recognise the sub-graph.
inline ov::Output<ov::Node> typed_scalar(ov::pass::NodeRegistry& reg,
                                         const ov::Output<ov::Node>& x,
                                         float value) {
    const auto& et = x.get_element_type();
    if (et.is_static() && et != ov::element::dynamic) {
        return reg.make<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{value});
    }
    auto c = reg.make<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{value});
    return reg.make<ov::op::v1::ConvertLike>(c, x);
}

}  // namespace detail
}  // namespace decompositions
}  // namespace ov
