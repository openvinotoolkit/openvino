// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/op/memory_access.hpp"

namespace ov {
namespace intel_cpu {

/**
* @interface BrgemmCopyB
* @brief The operation for data repacking of Brgemm with input non-fp32 precisions
* @ingroup snippets
*/
class BrgemmCopyB : public ngraph::snippets::op::MemoryAccess {
public:
    OPENVINO_OP("BrgemmCopyB", "SnippetsOpset", MemoryAccess);
    BrgemmCopyB(const Output<Node>& x, const element::Type src_type, const bool with_comp = false,
                  const size_t offset_in = 0lu, const size_t offset_out0 = 0lu, const size_t offset_out1 = 0lu);
    BrgemmCopyB() = default;

    size_t get_offset_in() const { return get_input_port_descriptor(0).m_offset; }
    size_t get_offset_out() const { return get_output_port_descriptor(0).m_offset; }
    size_t get_offset_comp() const { return get_output_port_descriptor(1).m_offset; }

    element::Type get_src_element_type() const { return m_src_type; }
    bool is_with_comp() const { return m_with_comp; }

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    bool has_evaluate() const override { return false; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    bool m_with_comp = false;
    element::Type m_src_type;  // src element type of the corresponding BRGEMM
};

} // namespace intel_cpu
} // namespace ov
