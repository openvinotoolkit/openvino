// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <openvino/op/op.hpp>

namespace ov {
namespace intel_cpu {
class NgramNode : public ov::op::Op {
public:
    OPENVINO_OP("NgramNode", "cpu_plugin_opset");

    NgramNode() = default;
    NgramNode(const ov::Output<Node>& embeddings, const ov::Output<Node>& batch_idces, const size_t k);
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    size_t get_k() const;

private:
    size_t m_k;
};
}   // namespace intel_cpu
}   // namespace ov
