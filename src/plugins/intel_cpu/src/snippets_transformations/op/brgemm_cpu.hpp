// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/op/brgemm.hpp"
#include "brgemm_copy_b.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @interface BrgemmCPU
 * @brief BrgemmCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class BrgemmCPU : public ngraph::snippets::op::Brgemm {
public:
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", ngraph::snippets::op::Brgemm);
    BrgemmCPU(const Output<Node>& A, const Output<Node>& B, bool transposed_a = false, bool transposed_b = false, const bool with_comp = false,
              const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_c = 0);
    BrgemmCPU(const Output<Node>& A, const Output<Node>& B, const Output<Node>& scratch,
              bool transposed_a = false, bool transposed_b = false, const bool with_comp = false,
              const size_t offset_a = 0, const size_t offset_b = 0, const size_t offset_scratch = 0, const size_t offset_c = 0);
    BrgemmCPU() = default;

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_offset_scratch() const { return get_input_port_descriptor(2).m_offset; }
    std::shared_ptr<BrgemmCopyB> get_brgemm_copy() const;

private:
    bool m_with_comp = false;  // compensations
};

} // namespace intel_cpu
} // namespace ov
