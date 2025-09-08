// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov::intel_cpu {

/**
 * @interface FusedMulAdd
 * @brief Fused Multiply Add
 * @ingroup snippets
 */
class FusedMulAdd : public ov::op::Op {
public:
    OPENVINO_OP("FusedMulAdd", "SnippetsOpset");

    FusedMulAdd() = default;
    FusedMulAdd(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c);

    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
    const ov::op::AutoBroadcastSpec& get_autob() const override;
};

}  // namespace ov::intel_cpu
