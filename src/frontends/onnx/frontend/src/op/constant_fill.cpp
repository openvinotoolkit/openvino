// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/constant_fill.hpp"

#include <onnx/onnx_pb.h>  // onnx types

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "onnx_common/utils.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector constant_fill(const Node& node) {
    Output<ngraph::Node> target_shape;
    const auto dtype =
        node.get_attribute_value<int64_t>("dtype", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    const auto ng_type = onnx_common::onnx_to_ng_data_type(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(dtype));
    const auto const_val_to_fill = node.get_attribute_as_constant<float>("value", 0.f, ng_type);
    const auto input_as_shape = node.get_attribute_value<int64_t>("input_as_shape", 1);
    if (input_as_shape == 1)  // use the first input as target shape
    {
        CHECK_VALID_NODE(node,
                         node.get_ng_inputs().size() > 0,
                         "The input which determines output shape was not provided");
        target_shape = node.get_ng_inputs().at(0);
        if (node.has_attribute("extra_shape")) {
            const auto extra_shape_const =
                node.get_attribute_as_constant<std::vector<int64_t>>("extra_shape", target_shape.get_element_type());
            target_shape = std::make_shared<default_opset::Concat>(OutputVector{target_shape, extra_shape_const}, 0);
        }
    } else  // use shape attribute as target shape
    {
        target_shape = node.get_attribute_as_constant<std::vector<int64_t>>("shape", ng_type);
    }

    return {std::make_shared<default_opset::Broadcast>(const_val_to_fill, target_shape)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
