// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using namespace ov::op::v0;
using namespace ov::op::v1;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ov::Shape;

ov::Output<ov::Node> rank(const ov::Output<ov::Node>& source) {
    return std::make_shared<Squeeze>(std::make_shared<v3::ShapeOf>(std::make_shared<v3::ShapeOf>(source)));
}

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector layer_normalization(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://github.com/onnx/onnx/blob/main/onnx/defs/nn/defs.cc#L2562:L2611
    const auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node,
                     num_inputs == 2 || num_inputs == 3,
                     "LayerNormalization expects 2 or 3 input tensors. Got: ",
                     num_inputs);
    CHECK_VALID_NODE(node,
                     node.get_outputs_size() == 1,
                     "LayerNormalization expects 1 output tensor to be used in a model, other configurations are used "
                     "for training and are not supported. Got: ",
                     node.get_outputs_size(),
                     " outputs.");

    auto default_stash_type_i = static_cast<int64_t>(TensorProto_DataType::TensorProto_DataType_FLOAT);
    int64_t stash_type_i = node.get_attribute_value<int64_t>("stash_type", default_stash_type_i);
    element::Type stash_type = common::get_ov_element_type(stash_type_i);

    ov::Output<ov::Node> data = inputs.at(0);
    element::Type original_type = data.get_element_type();
    bool needs_type_casting = stash_type != original_type;

    if (needs_type_casting)
        data = std::make_shared<Convert>(data, stash_type);

    float epsilon = node.get_attribute_value<float>("epsilon", 1e-5f);

    auto axis = node.get_attribute_value<std::int64_t>("axis", -1);
    // ONNX operator semantics says that `axis` attribute points to the first normalization dimension. We have to
    // figure out all dimensions for normalization:
    // axis < 0:  range(axis, 0)     Example: axis = -2; Axes: [-2, -1]
    // axis >= 0: range(axis, rank)  Example: axis = 2, rank = 4; Axes: [2, 3]
    auto axes =
        std::make_shared<v4::Range>(Constant::create(element::i64, {}, {axis}),
                                    (axis < 0 ? Constant::create(element::i64, {}, {0})->output(0) : rank(data)),
                                    Constant::create(element::i64, {}, {1}),
                                    element::i64);

    const auto normalize_variance = true;
    ov::Output<ov::Node> normalized =
        std::make_shared<v6::MVN>(data, axes, normalize_variance, epsilon, MVNEpsMode::INSIDE_SQRT);

    if (needs_type_casting)
        normalized = std::make_shared<ConvertLike>(normalized, inputs.at(0));

    auto scaled = std::make_shared<Multiply>(normalized, inputs.at(1));
    auto biased = (num_inputs == 3 ? std::make_shared<Add>(scaled, inputs.at(2))->output(0) : scaled->output(0));
    return ov::OutputVector{biased};
}

ONNX_OP("LayerNormalization", OPSET_SINCE(1), ai_onnx::opset_1::layer_normalization);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
