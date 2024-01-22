// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/constant_fill.hpp"

#include <onnx/onnx_pb.h>  // onnx types

#include "exceptions.hpp"
#include "onnx_common/utils.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"

using namespace ov::op;
using namespace ov::frontend::onnx::common;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector constant_fill(const Node& node) {
    Output<ov::Node> target_shape;
    const auto dtype = node.get_attribute_value<int64_t>("dtype", static_cast<int64_t>(TensorProto_DataType_FLOAT));
    const auto ng_type = onnx_to_ov_data_type(static_cast<TensorProto_DataType>(dtype));
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
            target_shape = std::make_shared<v0::Concat>(OutputVector{target_shape, extra_shape_const}, 0);
        }
    } else  // use shape attribute as target shape
    {
        target_shape = node.get_attribute_as_constant<std::vector<int64_t>>("shape", ng_type);
    }

    return {std::make_shared<v3::Broadcast>(const_val_to_fill, target_shape)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
