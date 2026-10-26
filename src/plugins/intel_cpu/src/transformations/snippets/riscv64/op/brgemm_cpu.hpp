// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::intel_cpu {

/**
 * @interface BrgemmCPU
 * @brief BrgemmCPU is a matrix multiplication with arbitrary row strides.
 * @ingroup snippets
 */
class BrgemmCPU : public snippets::op::Brgemm {
public:
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", snippets::op::Brgemm);

    BrgemmCPU(const Output<Node>& A,
              const Output<Node>& B,
              const PortDescriptor& desc_a,
              const PortDescriptor& desc_b,
              const PortDescriptor& desc_c,
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {});
    BrgemmCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
    static void validate_element_types(const ov::element::Type& type_a, const ov::element::Type& type_b);
};

}  // namespace ov::intel_cpu
