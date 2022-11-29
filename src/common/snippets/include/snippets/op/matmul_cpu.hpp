// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/matmul.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface LoadConvertSaturation
 * @brief Fused operation to represent computations equal to consecutive Load and ConvertSaturation operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class MatMulCPU : public ngraph::op::v0::MatMul {
public:
    OPENVINO_OP("MatMulCPU", "SnippetsOpset", ngraph::op::v0::MatMul);
    MatMulCPU(const Output<Node>& A, const Output<Node>& B, size_t offset_a = 0, size_t offset_b = 0, size_t offset_c = 0);
    MatMulCPU() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }

    size_t get_offset_a() const { return m_offset_a; }
    size_t get_offset_b() const { return m_offset_b; }
    size_t get_offset_c() const { return m_offset_c; }

    void set_offset_a(size_t offset) { m_offset_a = offset; }
    void set_offset_b(size_t offset) { m_offset_b = offset; }
    void set_offset_c(size_t offset) { m_offset_c = offset; }

private:
    MatMulCPU(const Output<Node>& A, const Output<Node>& B, std::vector<size_t> output_layout, size_t offset_a = 0, size_t offset_b = 0, size_t offset_c = 0);
    std::vector<size_t> m_output_layout;
    size_t m_offset_a = 0lu;
    size_t m_offset_b = 0lu;
    size_t m_offset_c = 0lu;
};

} // namespace op
} // namespace snippets
} // namespace ngraph