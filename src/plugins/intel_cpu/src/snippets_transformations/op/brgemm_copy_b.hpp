// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "snippets/op/memory_access.hpp"

namespace ov {
namespace intel_cpu {

/**
* @interface BrgemmCopyBBase
* @brief The base class with the common interface for data repacking of Brgemm with input non-fp32 precisions
* @ingroup snippets
*/
class BrgemmCopyBBase : public ngraph::snippets::op::MemoryAccess {
public:
    OPENVINO_OP("BrgemmCopyBBase", "SnippetsOpset", MemoryAccess);
    BrgemmCopyBBase() = default;

    size_t get_offset_in() const { return get_input_port_descriptor(0).m_offset; }
    size_t get_offset_out() const { return get_output_port_descriptor(0).m_offset; }

    element::Type get_src_element_type() const { return m_src_type; }

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    bool has_evaluate() const override { return false; }

protected:
    BrgemmCopyBBase(const Output<Node>& x, const element::Type src_type,
                    const size_t offset_in = 0lu, const size_t offset_out = 0lu);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override { return nullptr; };

    element::Type m_src_type;  // src element type of the corresponding BRGEMM (first input)
};

/**
* @interface BrgemmCopyB
* @brief The operation for data repacking of Brgemm with input non-fp32 precisions without compensations (doesn't have 2nd output)
* @ingroup snippets
*/
class BrgemmCopyB : public BrgemmCopyBBase {
public:
    OPENVINO_OP("BrgemmCopyB", "SnippetsOpset", BrgemmCopyBBase);
    BrgemmCopyB(const Output<Node>& x, const element::Type src_type,
                const size_t offset_in = 0lu, const size_t offset_out = 0lu);
    BrgemmCopyB() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

/**
* @interface BrgemmCopyBWithCompensations
* @brief The operation for data repacking of Brgemm with input non-fp32 precisions with compensations (has 2nd output)
* @ingroup snippets
*/
class BrgemmCopyBWithCompensations : public BrgemmCopyBBase {
public:
    OPENVINO_OP(" BrgemmCopyBWithCompensations", "SnippetsOpset", BrgemmCopyBBase);
    BrgemmCopyBWithCompensations(const Output<Node>& x, const element::Type src_type,
                                 const size_t offset_in = 0lu, const size_t offset_out0 = 0lu, const size_t offset_out1 = 0lu);
    BrgemmCopyBWithCompensations() = default;

    size_t get_offset_comp() const { return get_output_port_descriptor(1).m_offset; }

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

} // namespace intel_cpu
} // namespace ov
