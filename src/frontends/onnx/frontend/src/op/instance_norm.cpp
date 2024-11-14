// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector instance_norm(const ov::frontend::onnx::Node& node) {
    ov::Output<ov::Node> data(node.get_ov_inputs().at(0));
    ov::Output<ov::Node> scale(node.get_ov_inputs().at(1));
    ov::Output<ov::Node> bias(node.get_ov_inputs().at(2));
    const ov::PartialShape& data_pshape = data.get_partial_shape();
    const ov::PartialShape& scale_pshape = scale.get_partial_shape();
    const ov::PartialShape& bias_pshape = bias.get_partial_shape();
    const float epsilon{node.get_attribute_value<float>("epsilon", 1e-5f)};

    ov::element::Type result_et;
    CHECK_VALID_NODE(node,
                     ov::element::Type::merge(result_et, data.get_element_type(), scale.get_element_type()),
                     "Element types for data and scale input do not match (data element type: ",
                     data.get_element_type(),
                     ", scale element type: ",
                     scale.get_element_type(),
                     ").");

    CHECK_VALID_NODE(node,
                     ov::element::Type::merge(result_et, data.get_element_type(), bias.get_element_type()),
                     "Element types for data and bias input do not match (data element type: ",
                     data.get_element_type(),
                     ", bias element type: ",
                     bias.get_element_type(),
                     ").");

    if (data_pshape.rank().is_static()) {
        CHECK_VALID_NODE(
            node,
            scale_pshape.is_dynamic() || (scale_pshape.rank().is_static() && scale_pshape.rank().get_length() == 1 &&
                                          data_pshape[1].compatible(scale_pshape[0])),
            "Scale input must be one dimensional vector of number of "
            "input data channels size.");

        CHECK_VALID_NODE(
            node,
            bias_pshape.is_dynamic() || (bias_pshape.rank().is_static() && bias_pshape.rank().get_length() == 1 &&
                                         data_pshape[1].compatible(bias_pshape[0])),
            "Bias input must be one dimensional vector of number of "
            "input data channels size.");
    }

    // all dimensions except spatial/feature
    const auto reduction_axes = common::get_monotonic_range_along_node_rank(data, 2);

    auto mvn = std::make_shared<v6::MVN>(data, reduction_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);

    const auto mvn_shape = std::make_shared<v3::ShapeOf>(mvn);
    const auto mvn_rank = std::make_shared<v3::ShapeOf>(mvn_shape);

    // scale * mvn + bias
    std::shared_ptr<ov::Node> result =
        std::make_shared<v1::Multiply>(mvn, reshape::reshape_channel_shaped_node_to_nchw(scale, mvn_rank));
    result = std::make_shared<v1::Add>(result, reshape::reshape_channel_shaped_node_to_nchw(bias, mvn_rank));

    return {result};
}

ONNX_OP("InstanceNormalization", OPSET_SINCE(1), ai_onnx::opset_1::instance_norm);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
