// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/decompositions/rms_norm.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"

namespace ov {
namespace decompositions {

namespace {
// Create a scalar constant typed as `x` so that pattern matchers in the
// corresponding fusion transformation (which expect a raw `Constant` or
// `Convert(Constant)`, not `ConvertLike`) can recognise the sub-graph.
ov::Output<ov::Node> typed_scalar(ov::pass::NodeRegistry& reg, const ov::Output<ov::Node>& x, float value) {
    const auto& et = x.get_element_type();
    if (et.is_static() && et != ov::element::dynamic) {
        return reg.make<ov::op::v0::Constant>(et, ov::Shape{}, std::vector<float>{value});
    }
    auto c = reg.make<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{value});
    return reg.make<ov::op::v1::ConvertLike>(c, x);
}
}  // namespace

ov::Output<ov::Node> rms_norm(ov::pass::NodeRegistry& reg,
                              const ov::Output<ov::Node>& x,
                              const ov::Output<ov::Node>& axes,
                              const ov::Output<ov::Node>& eps,
                              const ov::Output<ov::Node>& scale) {
    // Decomposition shape:
    //   y = x * Power(Sqrt(ReduceMean(x^2, axes) + eps), -1) [* scale]
    // This exact graph is recognised by ov::pass::RMSFusion.
    auto squared = reg.make<ov::op::v1::Power>(x, typed_scalar(reg, x, 2.0f));
    auto mean = reg.make<ov::op::v1::ReduceMean>(squared, axes, /*keep_dims=*/true);
    auto mean_plus_eps = reg.make<ov::op::v1::Add>(mean, eps);
    auto sqrt = reg.make<ov::op::v0::Sqrt>(mean_plus_eps);
    auto inv_rms = reg.make<ov::op::v1::Power>(sqrt, typed_scalar(reg, x, -1.0f));

    ov::Output<ov::Node> result = reg.make<ov::op::v1::Multiply>(x, inv_rms);
    if (scale.get_node_shared_ptr()) {
        result = reg.make<ov::op::v1::Multiply>(scale, result);
    }
    return result;
}

}  // namespace decompositions
}  // namespace ov
