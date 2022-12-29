// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "memory_access.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Brgemm
 * @brief Brgemm is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 * @ingroup snippets
 */
class Brgemm : public MemoryAccess {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset", MemoryAccess);
    Brgemm(const Output<Node>& A, const Output<Node>& B, bool transposed_a = false, bool transposed_b = false,
           const size_t offset_a = 0lu, const size_t offset_b = 0lu, const size_t offset_c = 0lu);
    Brgemm() = default;

    bool transposed_a() const { return m_transposed_a; }
    bool transposed_b() const { return m_transposed_b; }

    size_t get_offset_a() const { return get_input_port_descriptor(0).m_offset; }
    size_t get_offset_b() const { return get_input_port_descriptor(1).m_offset; }
    size_t get_offset_c() const { return get_output_port_descriptor(0).m_offset; }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }

protected:
    ov::element::Type get_output_type() const;
    ov::PartialShape get_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) const;

    bool m_transposed_a;
    bool m_transposed_b;
};

} // namespace op
} // namespace snippets
} // namespace ngraph