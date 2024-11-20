// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"

#include "openvino/op/group_query_attention.hpp"


using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {


namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    // Replace ONNX specific null inputs to GQA null inputs
    auto onnx_op_inputs = node.get_ov_inputs();
    OutputVector ov_op_inputs;
    ov_op_inputs.reserve(onnx_op_inputs.size());
    for(const auto& input: onnx_op_inputs) {
        ov_op_inputs.push_back(ov::op::util::is_null(input) ? GroupQueryAttentionExtension::null() : input);
    }

    return std::make_shared<GroupQueryAttentionExtension>(
        ov_op_inputs,
        static_cast<int>(node.get_attribute_value<int64_t>("num_heads")),
        static_cast<int>(node.get_attribute_value<int64_t>("kv_num_heads")),
        static_cast<bool>(node.get_attribute_value<int64_t>("rotary_interleaved", 0))
    )->outputs();
}

ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
