// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/visibility.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {

/// \brief SequenceMark serves to mark places that require a sequence type propagation.
/// This class represents list or tuple constructs in frameworks like PyTorch, ONNX, etc.
/// It holds a collection of tensors that form a sequence.
class FRONTEND_API SequenceMark : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("SequenceMark", "util", ov::op::util::FrameworkNode);

    SequenceMark(const ov::OutputVector& inputs);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override;

    size_t size() const;
    bool empty() const;
    ov::Output<ov::Node> get_element(size_t index) const;
    ov::OutputVector get_sequence() const;
};

}  // namespace frontend
}  // namespace ov
