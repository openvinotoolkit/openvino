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

ov::OutputVector sequence_empty(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 0, 0);
    auto sequence = std::make_shared<ov::frontend::SequenceMark>(ov::OutputVector{});
    ov::op::util::FrameworkNodeAttrs attrs;
    attrs.set_type_name("SequenceEmpty");
    sequence->set_attrs(attrs);
    return {sequence};
}

ONNX_OP("SequenceEmpty", OPSET_SINCE(1), ai_onnx::opset_11::sequence_empty);

}  // namespace opset_11
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
