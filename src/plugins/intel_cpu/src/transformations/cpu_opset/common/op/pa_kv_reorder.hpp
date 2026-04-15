// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_cpu::op {

class PaKVReorder : public ov::op::Op {
public:
    OPENVINO_OP("PaKVReorder", "cpu_plugin_opset");

    PaKVReorder() = default;

    PaKVReorder(const Output<Node>& key_cache,
                const Output<Node>& value_cache,
                const Output<Node>& block_indices,
                const Output<Node>& block_indices_begins,
                const Output<Node>& block_update_indices,
                const Output<Node>& block_update_indices_begins);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_cpu::op
