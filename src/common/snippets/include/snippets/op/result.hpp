// Copyright (C) 2018-2025 Intel Corporation
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
#include "snippets/utils/utils.hpp"

namespace ov::snippets::op {

// Result op in snippets has multiple inputs connected to specific loops' output.
class Result : public ov::op::Op {
public:
    OPENVINO_OP("SnippetsResult", "SnippetsOpset");
    Result() = default;
    explicit Result(const OutputVector& arguments);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace ov::snippets::op
