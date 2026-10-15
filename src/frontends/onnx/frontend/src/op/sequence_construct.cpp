// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/sequence_mark.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_11 {

ov::OutputVector sequence_construct(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 1);
    const auto& inputs = node.get_ov_inputs();

    // Create SequenceMark from inputs
    auto sequence = std::make_shared<ov::frontend::SequenceMark>(inputs);
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("SequenceConstruct");
    sequence->set_attrs(attrs);
    return {sequence};
}

ONNX_OP("SequenceConstruct", OPSET_SINCE(1), ai_onnx::opset_11::sequence_construct);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
