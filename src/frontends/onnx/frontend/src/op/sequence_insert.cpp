// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_insert.hpp"

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

ov::OutputVector sequence_insert(const ov::frontend::onnx::Node& node) {
    constexpr auto min_inputs = 2;
    constexpr auto max_inputs = 3;
    common::default_op_checks(node, min_inputs, max_inputs);

    const auto& inputs = node.get_ov_inputs();
    const auto& sequence_input = inputs[0];
    const auto& to_insert = inputs[1];

    // Create a SequenceInsert helper op that will be resolved by a transformation pass
    if (inputs.size() == 3) {
        return {std::make_shared<ov::frontend::SequenceInsert>(sequence_input, to_insert, inputs[2])};
    }
    return {std::make_shared<ov::frontend::SequenceInsert>(sequence_input, to_insert)};
}

ONNX_OP("SequenceInsert", OPSET_SINCE(1), ai_onnx::opset_11::sequence_insert);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
