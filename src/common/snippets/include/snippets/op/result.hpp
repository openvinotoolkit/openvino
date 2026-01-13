// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/result.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::op {

// Snippets result is used only in snippets lowering pipeline as output of subgraph.
// This op is needed becuase there could be multiple loop bodies(such as first, main and tail) in LIR,
// all need write to the same output but at different offsets. Then snippets result should have
// multiple inputs connected to all loops. The correct and full connections are needed in passes like AssignRegisters.
// Note that Result tensor is same as main input tensor (precision, names etc),
// the other inputs tensor properties are ignored and hidden from model output perspective.
class Result : public ov::op::v0::Result {
public:
    OPENVINO_OP("Result", "SnippetsOpset", ov::op::v0::Result);
    Result() = default;
    explicit Result(const Output<Node>& main, const OutputVector& nodes = {});

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
};

}  // namespace ov::snippets::op
