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
 * @interface Brgemm
 * @brief Brgemm is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 * @ingroup snippets
 */
class Brgemm : public ngraph::op::v0::MatMul {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset", ngraph::op::v0::MatMul);
    Brgemm(const Output<Node>& A, const Output<Node>& B, const size_t offset_a = 0lu, const size_t offset_b = 0lu, const size_t offset_c = 0lu);
    Brgemm() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }

    size_t get_offset_a() const { return m_offset_a; }
    size_t get_offset_b() const { return m_offset_b; }
    size_t get_offset_c() const { return m_offset_c; }

    void set_offset_a(const size_t offset) { m_offset_a = offset; }
    void set_offset_b(const size_t offset) { m_offset_b = offset; }
    void set_offset_c(const size_t offset) { m_offset_c = offset; }

private:
    size_t m_offset_a = 0lu;  // offset for first input
    size_t m_offset_b = 0lu;  // offset for second input
    size_t m_offset_c = 0lu;  // offset for output
};

} // namespace op
} // namespace snippets
} // namespace ngraph