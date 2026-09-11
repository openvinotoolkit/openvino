// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector gather(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ng_inputs{node.get_ov_inputs()};
    auto data = ng_inputs.at(0);
    auto indices = ng_inputs.at(1);
    auto axis = node.get_attribute_value<int64_t>("axis", 0);

    const auto& data_shape = data.get_partial_shape();
    if (data_shape.rank().is_static()) {
        const auto data_rank = data_shape.rank().get_length();
        const int64_t norm_axis = axis < 0 ? axis + data_rank : axis;
        CHECK_VALID_NODE(node,
                         norm_axis >= 0 && norm_axis < data_rank,
                         "Gather attribute 'axis' = ",
                         axis,
                         " is out of range for data of rank ",
                         data_rank);
        if (data_shape[norm_axis].is_static()) {
            const auto indices_const = ov::as_type_ptr<ov::op::v0::Constant>(indices.get_node_shared_ptr());
            if (indices_const != nullptr) {
                const int64_t axis_size = data_shape[norm_axis].get_length();
                for (const int64_t idx : indices_const->cast_vector<int64_t>()) {
                    CHECK_VALID_NODE(node,
                                     idx >= -axis_size && idx < axis_size,
                                     "Gather index ",
                                     idx,
                                     " is out of bounds for axis ",
                                     norm_axis,
                                     " of size ",
                                     axis_size,
                                     " (allowed range [",
                                     -axis_size,
                                     ", ",
                                     axis_size - 1,
                                     "])");
                }
            }
        }
    }

    return {std::make_shared<ov::op::v8::Gather>(data,
                                                 indices,
                                                 ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {axis}))};
}

ONNX_OP("Gather", OPSET_SINCE(1), ai_onnx::opset_1::gather);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
