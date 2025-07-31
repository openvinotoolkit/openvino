// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
// #include "openvino/op/util/unary_elementwise_arithmetic.hpp"

namespace ov {
namespace op {
namespace v5 {

   void get_rel_pos_freq(float* pe_freq, int embed_dims, int pos_dim);
    
    void attention_value_computation(
        int bs, int head, int seq_len, int embed_dims, int pos_dim,
        int num_head_idx, const float* query, const float* key,
        const int* index, const float* pe_freq, const float* pe_phase, float* output
    );

    void attention_weight_computation(
        int bs, int head, int seq_len, int embed_dims, int pos_dim,
        int num_head_idx, const float* query, const float* key,
        const int* index, const float* pe_freq, const float* pe_phase, float* output
    );

class OPENVINO_API RotRPEAttentionWeightWithIndexComputation : public ov::op::Op {
public:
    OPENVINO_OP("RotRPEAttentionWeightWithIndexComputation", "opset5", op::Op);

    /// \brief Constructs a round operation.
    RotRPEAttentionWeightWithIndexComputation() = default;

    /// \brief Constructs a round operation.
    RotRPEAttentionWeightWithIndexComputation(const Output<Node>& input0,
                                              const Output<Node>& input1,
                                              const Output<Node>& input2,
                                              const Output<Node>& input3);

    // bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};


class OPENVINO_API RotRPEProjectValueWithIndexComputation : public ov::op::Op {
public:
    OPENVINO_OP("RotRPEProjectValueWithIndexComputation", "opset5", op::Op);

    /// \brief Constructs a round operation.
    RotRPEProjectValueWithIndexComputation() = default;

    /// \brief Constructs a round operation.
    RotRPEProjectValueWithIndexComputation(const Output<Node>& input0,
                                              const Output<Node>& input1,
                                              const Output<Node>& input2,
                                              const Output<Node>& input3);

    // bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};


}  // namespace v5
}  // namespace op
}  // namespace ov
