// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::snippets::op {

/**
 * @interface HorizonMax
 * @brief The operation calculates a horizon maximum of a vector register
 * @ingroup snippets
 */
class HorizonMax : public ov::op::Op {
public:
    OPENVINO_OP("HorizonMax", "SnippetsOpset");

    explicit HorizonMax(const Output<Node>& x);
    HorizonMax() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op
