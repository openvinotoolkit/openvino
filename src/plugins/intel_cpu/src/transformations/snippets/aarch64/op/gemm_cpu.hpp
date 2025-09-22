// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
            const PortDescriptor& desc_a,
            const PortDescriptor& desc_b,
            const PortDescriptor& desc_c,
            const std::vector<size_t>& layout_a = {},
            const std::vector<size_t>& layout_b = {},
            const std::vector<size_t>& layout_c = {});
    GemmCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
    static void validate_element_type(const ov::element::Type& type_0, const ov::element::Type& type_1);
};
}  // namespace ov::intel_cpu::aarch64
