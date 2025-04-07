// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/node.hpp>
#include <openvino/op/op.hpp>

namespace ov::intel_cpu {
/**
 * The operation flattens embedding tensor of token vectors and traverse it by a sliding window of size times the
 * original embedding sizes. Inputs:
 *     1. Embedding vectors of type T1 - shape [N, m], where N - number of tokens, m - embedding size. Required
 *     2. Indices of type T2 - shape [N, 2]. Contains pairs <batch_idx;idx> for the corresponding tokens. This op uses
 * only batch indices. Required Outputs:
 *     1. New embedding vector of type T1 and of shape [N, m * k], where k - operation attribute.
 * Types:
 *     T1 - only FP32 is supported
 *     T2 - I32 and I64 are supported
 */
class NgramNode : public ov::op::Op {
public:
    OPENVINO_OP("Ngram", "cpu_plugin_opset");

    NgramNode() = default;
    NgramNode(const ov::Output<Node>& embeddings, const ov::Output<Node>& batch_idces, const size_t k);
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    size_t get_k() const;

private:
    size_t m_k;
};
}  // namespace ov::intel_cpu
