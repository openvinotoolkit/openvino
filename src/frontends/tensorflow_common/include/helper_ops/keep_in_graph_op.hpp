// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "helper_ops/internal_operation.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class KeepInGraphOp : public InternalOperation {
public:
    OPENVINO_OP("KeepInGraphOp", "ov::frontend::tensorflow", InternalOperation);

    KeepInGraphOp(const std::string& op_type_name,
                  const OutputVector& inputs,
                  const std::shared_ptr<DecoderBase>& decoder = nullptr)
        : InternalOperation(decoder, inputs, 1, op_type_name),
          m_op_type_name(op_type_name) {}

    void validate_and_infer_types() override {
        set_output_type(0, ov::element::dynamic, ov::PartialShape::dynamic());
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto keep_in_graph_node = std::make_shared<KeepInGraphOp>(m_op_type_name, inputs, m_decoder);
        keep_in_graph_node->set_attrs(get_attrs());
        return keep_in_graph_node;
    }

private:
    std::string m_op_type_name;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
