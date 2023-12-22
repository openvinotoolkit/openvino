// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/instance_norm.hpp"

#include <cstddef>
#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/partial_shape.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector instance_norm(const Node& node) {
    Output<ngraph::Node> data(node.get_ng_inputs().at(0));
    Output<ngraph::Node> scale(node.get_ng_inputs().at(1));
    Output<ngraph::Node> bias(node.get_ng_inputs().at(2));
    const PartialShape& data_pshape = data.get_partial_shape();
    const PartialShape& scale_pshape = scale.get_partial_shape();
    const PartialShape& bias_pshape = bias.get_partial_shape();
    const float epsilon{node.get_attribute_value<float>("epsilon", 1e-5f)};

    element::Type result_et;
    CHECK_VALID_NODE(node,
                     element::Type::merge(result_et, data.get_element_type(), scale.get_element_type()),
                     "Element types for data and scale input do not match (data element type: ",
                     data.get_element_type(),
                     ", scale element type: ",
                     scale.get_element_type(),
                     ").");

    CHECK_VALID_NODE(node,
                     element::Type::merge(result_et, data.get_element_type(), bias.get_element_type()),
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

    auto mvn =
        std::make_shared<default_opset::MVN>(data, reduction_axes, true, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);

    const auto mvn_shape = std::make_shared<default_opset::ShapeOf>(mvn);
    const auto mvn_rank = std::make_shared<default_opset::ShapeOf>(mvn_shape);

    // scale * mvn + bias
    std::shared_ptr<ngraph::Node> result =
        std::make_shared<default_opset::Multiply>(mvn, reshape::reshape_channel_shaped_node_to_nchw(scale, mvn_rank));
    result = std::make_shared<default_opset::Add>(result, reshape::reshape_channel_shaped_node_to_nchw(bias, mvn_rank));

    return {result};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
