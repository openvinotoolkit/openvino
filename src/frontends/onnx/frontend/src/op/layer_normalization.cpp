// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/layer_normalization.hpp"

#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using namespace ov::op::v0;
using namespace ov::op::v1;
using ::ONNX_NAMESPACE::TensorProto_DataType;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector layer_normalization(const ov::frontend::onnx::Node& node) {
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

    auto axis_value = node.get_attribute_value<std::int64_t>("axis", -1);
    float epsilon = node.get_attribute_value<float>("epsilon", 1e-5);
    auto axis = Constant::create(element::i64, {1}, {axis_value});
    const auto normalize_variance = true;
    ov::Output<ov::Node> normalized =
        std::make_shared<v6::MVN>(data, axis, normalize_variance, epsilon, MVNEpsMode::INSIDE_SQRT);

    if (needs_type_casting)
        normalized = std::make_shared<ConvertLike>(normalized, inputs.at(0));

    const auto& scale = inputs.at(1);
    auto scaled = std::make_shared<Multiply>(normalized, scale);
    auto biased = (num_inputs == 3 ? std::make_shared<Add>(scaled, inputs.at(2))->output(0) : scaled->output(0));
    return ov::OutputVector{biased};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
