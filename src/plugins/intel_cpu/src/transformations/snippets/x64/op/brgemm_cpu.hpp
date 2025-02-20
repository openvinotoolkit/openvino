// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "brgemm_copy_b.hpp"
#include "brgemm_utils.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::intel_cpu {

/**
 * @interface BrgemmCPU
 * @brief BrgemmCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class BrgemmCPU : public snippets::op::Brgemm {
public:
    using BRGEMM_TYPE = brgemm_utils::BRGEMM_TYPE;
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", snippets::op::Brgemm);

    BrgemmCPU(const Output<Node>& A,
              const Output<Node>& B,
              BRGEMM_TYPE type,
              const size_t offset_a = 0,
              const size_t offset_b = 0,
              const size_t offset_c = 0,
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {});
    BrgemmCPU(const Output<Node>& A,
              const Output<Node>& B,
              const Output<Node>& scratch,
              BRGEMM_TYPE type,
              const size_t offset_a = 0,
              const size_t offset_b = 0,
              const size_t offset_scratch = 0,
              const size_t offset_c = 0,
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {});
    BrgemmCPU(const Output<Node>& A,
              const Output<Node>& B,
              BRGEMM_TYPE type,
              const PortDescriptor& desc_a,
              const PortDescriptor& desc_b,
              const PortDescriptor& desc_c,
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {});
    BrgemmCPU(const Output<Node>& A,
              const Output<Node>& B,
              const Output<Node>& scratch,
              BRGEMM_TYPE type,
              const PortDescriptor& desc_a,
              const PortDescriptor& desc_b,
              const PortDescriptor& desc_scratch,
              const PortDescriptor& desc_c,
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {});
    BrgemmCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    BRGEMM_TYPE get_type() const {
        return m_type;
    }

    size_t get_offset_scratch() const;

    bool visit_attributes(AttributeVisitor& visitor) override;

    constexpr static size_t SCRATCH_BYTE_SIZE = 32 * 1024;

private:
    void custom_constructor_validate_and_infer_types(const std::vector<size_t>& layout_a,
                                                     const std::vector<size_t>& layout_b,
                                                     const std::vector<size_t>& layout_c);
    void validate_with_scratchpad() const;
    void validate_inputs() const;

    BRGEMM_TYPE m_type = BRGEMM_TYPE::STAND_ALONE;
};
}  // namespace ov::intel_cpu
