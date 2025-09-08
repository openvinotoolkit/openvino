// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace util {
bool is_null(const ov::Node* node);
bool is_null(const std::shared_ptr<ov::Node>& node);
bool is_null(const Output<ov::Node>& output);
}  // namespace util
}  // namespace op

namespace frontend {
namespace onnx {
/// \brief Represents a missing optional input or output of an ONNX node
///
/// Some ONNX operators have inputs or outputs that are marked as optional,
/// which means that a referring node MAY forgo providing values for such inputs
/// or computing these outputs.
/// An empty string is used in place of a name of such input or output.
///
/// More:
/// https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
class NullNode : public ov::op::Op {
public:
    OPENVINO_OP("NullNode");
    NullNode() {
        set_output_size(1);
    }

    virtual std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
