// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sqrt.hpp"
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
    common::default_op_checks(node, 2);

    const auto inputs = node.get_ov_inputs();
    auto X = inputs[0];
    const auto scale = inputs[1];

    const float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);
    const int64_t axis = node.get_attribute_value<int64_t>("axis", -1);

    const int64_t default_stash_type = static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT);
    const int64_t stash_type_i = node.get_attribute_value<int64_t>("stash_type", default_stash_type);
    const ov::element::Type stash_type = common::get_ov_element_type(stash_type_i);

    const ov::element::Type original_type = X.get_element_type();
    const bool needs_type_casting = stash_type != original_type;

    if (needs_type_casting) {
        X = std::make_shared<v0::Convert>(X, stash_type);
    }

    auto rank = std::make_shared<v0::Squeeze>(std::make_shared<v3::ShapeOf>(std::make_shared<v3::ShapeOf>(X)));
    auto axes = std::make_shared<v4::Range>(
        v0::Constant::create(ov::element::i64, {}, {axis}),
        (axis < 0 ? v0::Constant::create(ov::element::i64, {}, {0})->output(0) : rank->output(0)),
        v0::Constant::create(ov::element::i64, {}, {1}),
        ov::element::i64);

    auto squared_X = std::make_shared<v1::Multiply>(X, X);
    auto mean = std::make_shared<v1::ReduceMean>(squared_X, axes, true);
    auto rms =
        std::make_shared<v0::Sqrt>(std::make_shared<v1::Add>(mean, v0::Constant::create(stash_type, {}, {epsilon})));

    ov::Output<ov::Node> normalized = std::make_shared<v1::Divide>(X, rms);

    normalized = std::make_shared<v1::ConvertLike>(normalized, inputs[1]);

    auto normalized_shape = std::make_shared<v0::ShapeOf>(normalized);
    auto sub_shape = std::make_shared<v8::Slice>(normalized_shape,
                                                 v0::Constant::create(ov::element::i64, {1}, {axis}),
                                                 v0::Constant::create(ov::element::i64, {1}, {INT_MAX}),
                                                 v0::Constant::create(ov::element::i64, {1}, {1}));

    auto normalized_rank = normalized.get_partial_shape().rank();
    auto scale_input = scale;
    auto scale_rank = scale_input.get_partial_shape().rank();

    if ((scale_rank.is_dynamic() && normalized_rank.is_dynamic()) ||
        ((scale_rank.is_static() && normalized_rank.is_static()) &&
         scale_rank.get_length() + ov::util::normalize_axis(axis, normalized_rank.get_length()) !=
             static_cast<size_t>(normalized_rank.get_length()))) {
        scale_input = std::make_shared<v1::Reshape>(scale_input, sub_shape, false);
    }

    auto result = std::make_shared<v1::Multiply>(normalized, scale_input);
    return {result->output(0)};
}

ONNX_OP("RMSNormalization", OPSET_SINCE(1), ai_onnx::opset_23::rms_normalization);

}  // namespace opset_23
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
