// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "brgemm_copy_b.hpp"
#include "brgemm_utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::intel_cpu::aarch64 {

/**
 * @interface GemmCPU
 * @brief GemmCPU is a matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class GemmCPU : public snippets::op::Brgemm {
public:
    OPENVINO_OP("GemmCPU", "SnippetsOpset", snippets::op::Brgemm);

    GemmCPU(const Output<Node>& A,
            const Output<Node>& B,
            const size_t offset_a = 0,
            const size_t offset_b = 0,
            const size_t offset_c = 0,
            const std::vector<size_t>& layout_a = {},
            const std::vector<size_t>& layout_b = {},
            const std::vector<size_t>& layout_c = {});
    GemmCPU(const Output<Node>& A,
            const Output<Node>& B,
            const PortDescriptor& desc_a,
            const PortDescriptor& desc_b,
            const PortDescriptor& desc_c,
            const std::vector<size_t>& layout_a = {},
            const std::vector<size_t>& layout_b = {},
            const std::vector<size_t>& layout_c = {});
    GemmCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
    void validate_with_scratchpad() const;
    void validate_inputs() const;
};
}  // namespace ov::intel_cpu::aarch64
