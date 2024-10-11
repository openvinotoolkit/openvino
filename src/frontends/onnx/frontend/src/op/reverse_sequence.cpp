// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse_sequence.hpp"

#include "core/operator_set.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/convert.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector reverse_sequence(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    const auto sequence_lengths = node.get_ov_inputs().at(1);
    // OpenVINO supports only int32 type of sequence_lengths
    const auto sequence_lengths_i32 = std::make_shared<v0::Convert>(node.get_ov_inputs().at(1), ov::element::i32);
    const auto data_rank = data.get_partial_shape().rank();

    const auto batch_axis = node.get_attribute_value<int64_t>("batch_axis", 1);
    const auto normalized_batch_axis = common::normalize_axis(node.get_description(), batch_axis, data_rank);
    const auto time_axis = node.get_attribute_value<int64_t>("time_axis", 0);
    const auto normalized_time_axis = common::normalize_axis(node.get_description(), time_axis, data_rank);

    FRONT_END_GENERAL_CHECK(normalized_batch_axis == 0 || normalized_batch_axis == 1,
                            "Allowed values of the 'batch_axis' attribute for ReverseSequence "
                            "operator are 0 and 1");

    FRONT_END_GENERAL_CHECK(normalized_time_axis == 0 || normalized_time_axis == 1,
                            "Allowed values of the 'time_axis' attribute for ReverseSequence "
                            "operator are 0 and 1");

    FRONT_END_GENERAL_CHECK(normalized_batch_axis != normalized_time_axis,
                            "'batch_axis' and 'time_axis' attributes of the ReverseSequence "
                            "operator can't point to the same dimension");

    return {
        std::make_shared<v0::ReverseSequence>(data, sequence_lengths_i32, normalized_batch_axis, normalized_time_axis)};
}

ONNX_OP("ReverseSequence", OPSET_SINCE(1), ai_onnx::opset_1::reverse_sequence);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
