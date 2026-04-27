// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ::ONNX_NAMESPACE::TensorProto_DataType;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_23 {

ov::OutputVector rms_normalization(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://onnx.ai/onnx/operators/onnx__RMSNormalization.html
    //   Y = Mul(Cast(X / Sqrt(ReduceMean(X*X, [axis,..,rank-1]) + epsilon), T), scale)
    // The inner inv-rms graph is built via the shared decomposition helper so that
    // ov::pass::RMSFusion can fuse it back into ov::op::internal::RMS in plugins.
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node,
                     inputs.size() == 2,
                     "RMSNormalization expects 2 input tensors (X, scale). Got: ",
                     inputs.size());

    const int64_t axis = node.get_attribute_value<int64_t>("axis", -1);
    const float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);
    const int64_t stash_type_i =
        node.get_attribute_value<int64_t>("stash_type",
                                          static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT));
    const ov::element::Type stash_type = common::get_ov_element_type(stash_type_i);

    ov::Output<ov::Node> x = inputs[0];
    const ov::Output<ov::Node>& scale = inputs[1];
    const ov::element::Type original_type = x.get_element_type();
    const bool needs_cast = stash_type != original_type;

    if (needs_cast) {
        x = std::make_shared<v0::Convert>(x, stash_type);
    }

    // normalized_axes = [axis, axis+1, ..., rank(X)-1]
    // For axis < 0, equivalent to Range(axis, 0, 1) which yields exactly the
    // negative tail (e.g. axis=-2 → [-2, -1]).
    const auto axes_start = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    const auto axes_step = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    ov::Output<ov::Node> axes_end;
    if (axis < 0) {
        axes_end = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    } else {
        // rank(x) = Squeeze(ShapeOf(ShapeOf(x)))
        auto x_shape = std::make_shared<v3::ShapeOf>(x, ov::element::i64);
        auto x_rank_1d = std::make_shared<v3::ShapeOf>(x_shape, ov::element::i64);
        axes_end = std::make_shared<v0::Squeeze>(x_rank_1d);
    }
    auto axes = std::make_shared<v4::Range>(axes_start, axes_end, axes_step, ov::element::i64);

    // Epsilon constant in the same precision as `x` (the helper's Add(ReduceMean, eps)
    // requires matching dtypes; RMSFusion expects raw Constant, not ConvertLike).
    const auto x_et = x.get_element_type();
    ov::Output<ov::Node> eps;
    if (x_et.is_static()) {
        eps = v0::Constant::create(x_et, ov::Shape{}, {epsilon});
    } else {
        const auto eps_f32 = v0::Constant::create(ov::element::f32, ov::Shape{}, {epsilon});
        eps = std::make_shared<v1::ConvertLike>(eps_f32, x);
    }

    // Build inv-rms via the shared helper (no scale here — scale is applied after the
    // cast back to the original type, per ONNX spec).
    ov::pass::NodeRegistry reg;
    ov::Output<ov::Node> normalized = ov::decompositions::rms_norm(reg, x, axes, eps);

    if (needs_cast) {
        normalized = std::make_shared<v1::ConvertLike>(normalized, inputs[0]);
    }

    return {std::make_shared<v1::Multiply>(normalized, scale)};
}

ONNX_OP("RMSNormalization", OPSET_SINCE(1), ai_onnx::opset_23::rms_normalization);

}  // namespace opset_23
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
