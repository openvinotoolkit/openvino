// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "memory_access.hpp"
#include "snippets/shape_inference/shape_inference.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface Brgemm
 * @brief Brgemm is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 * @ingroup snippets
 */
class Brgemm : virtual public modifier::MemoryAccess, public ov::op::Op {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset");
    Brgemm(const Output<Node>& A, const Output<Node>& B,
           const size_t offset_a = 0lu, const size_t offset_b = 0lu, const size_t offset_c = 0lu,
           std::vector<size_t> layout_a = {}, std::vector<size_t> layout_b = {}, std::vector<size_t> layout_c = {},
           size_t blk_size_m = 0, size_t blk_size_k = 0, size_t blk_size_n = 0);
    Brgemm(const Output<Node>& A, const Output<Node>& B,
           const PortDescriptor& desc_a, const PortDescriptor& desc_b, const PortDescriptor& desc_c,
           std::vector<size_t> layout_a = {}, std::vector<size_t> layout_b = {}, std::vector<size_t> layout_c = {},
           size_t blk_size_m = 0, size_t blk_size_k = 0, size_t blk_size_n = 0);
    Brgemm() = default;

    size_t get_offset_a() const { return get_input_offset(0); }
    size_t get_offset_b() const { return get_input_offset(1); }
    size_t get_offset_c() const { return get_output_offset(0); }

    size_t get_m_block_size() const { return m_M_blk; }
    size_t get_k_block_size() const { return m_K_blk; }
    size_t get_n_block_size() const { return m_N_blk; }
    float get_beta() const { return m_beta; }

    void set_m_block_size(size_t block_size) { m_M_blk = block_size; }
    void set_k_block_size(size_t block_size) { m_K_blk = block_size; }
    void set_n_block_size(size_t block_size) { m_N_blk = block_size; }
    void set_beta(float beta) { m_beta = beta; }

    static ov::element::Type get_output_type(const ov::element::Type& in_type0, const ov::element::Type& in_type1);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }
    bool visit_attributes(AttributeVisitor& visitor) override;

protected:
    ov::element::Type get_output_type() const;
    std::vector<ov::PartialShape> get_planar_input_shapes(const std::vector<ov::Input<ov::Node>>& inputs) const;
    ov::PartialShape infer_output_partial_shape(const std::vector<ov::PartialShape>& input_shapes) const;
    ov::PartialShape get_planar_output_shape(const ov::PartialShape& output_shape) const;
    void set_block_size_values(size_t blk_size_m, size_t blk_size_k, size_t blk_size_n);
    size_t m_M_blk = 0;
    size_t m_K_blk = 0;
    size_t m_N_blk = 0;
    float m_beta = 0.f;

private:
    void custom_constructor_validate_and_infer_types(std::vector<size_t> layout_a, std::vector<size_t> layout_b, std::vector<size_t> layout_c);
};

} // namespace op
} // namespace snippets
} // namespace ov