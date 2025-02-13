// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/null.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {

namespace opset_1 {
ov::OutputVector group_query_attention(const ov::frontend::onnx::Node& node) {
    // At least given "query" and "seqlens_k"
    common::default_op_checks(node, 2);

    const auto onnx_op_inputs = node.get_ov_inputs();
    const auto num_heads = node.get_attribute_value<int64_t>("num_heads");
    const auto kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads");
    const auto scale = node.get_attribute_value<float>("scale", 0.0f);
    const auto do_rotary = node.get_attribute_value<int64_t>("do_rotary", 0);
    const auto rotary_interleaved = node.get_attribute_value<int64_t>("rotary_interleaved", 0);

    OutputVector ov_op_inputs;
    ov_op_inputs.reserve(onnx_op_inputs.size());
    for (const auto& input : onnx_op_inputs) {
        ov_op_inputs.push_back(ov::op::util::is_null(input) ? std::make_shared<v15::Null>() : input);
    }
    // total_sequence_length is not used currently in OV GQA
    ov_op_inputs[6] = std::make_shared<v15::Null>();
    return std::make_shared<v15::GroupQueryAttention>(ov_op_inputs,
                                                      num_heads,
                                                      kv_num_heads,
                                                      scale,
                                                      do_rotary,
                                                      rotary_interleaved)
        ->outputs();
}

ONNX_OP("GroupQueryAttention", OPSET_SINCE(1), com_microsoft::opset_1::group_query_attention, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
