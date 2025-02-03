// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v15 {

/// \brief Represents a missing optional input or output of an ONNX node
///
/// Some ONNX operators have inputs or outputs that are marked as optional,
/// which means that a referring node MAY forgo providing values for such inputs
/// or computing these outputs.
/// An empty string is used in place of a name of such input or output.
///
/// More:
/// https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
class OPENVINO_API Null : public Op {
public:
    OPENVINO_OP("Null", "opset15", op::Op);
    Null() {
        set_output_size(1);
    }

    static bool is_null(const ov::Node* node) {
        return ov::as_type<const ov::op::v15::Null>(node) != nullptr;
    }

    static bool is_null(const std::shared_ptr<ov::Node>& node) {
        return is_null(node.get());
    }

    static bool is_null(const Output<ov::Node>& output) {
        return is_null(output.get_node());
    }

    virtual std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<ov::op::v15::Null>();
    }
};
}  // namespace v15
}  // namespace op
}  // namespace ov
