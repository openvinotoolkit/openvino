// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "brgemm_utils.hpp"
#include "gemm_cpu.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"

namespace ov::intel_cpu {

/**
 * @interface BrgemmCPU
 * @brief BrgemmCPU is a batch-reduced matrix multiplication with the support of arbitrary strides between matrices rows
 *        with support of several precisions on plugin level
 * @ingroup snippets
 */
class BrgemmCPU : public GemmCPU {
public:
    using BRGEMM_TYPE = brgemm_utils::BRGEMM_TYPE;
    OPENVINO_OP("BrgemmCPU", "SnippetsOpset", GemmCPU);

    BrgemmCPU(const Output<Node>& A,
              const Output<Node>& B,
              size_t iter_count,
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
              size_t iter_count,
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
              size_t iter_count,
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
              size_t iter_count,
              BRGEMM_TYPE type,
              const PortDescriptor& desc_a,
              const PortDescriptor& desc_b,
              const PortDescriptor& desc_scratch,
              const PortDescriptor& desc_c,
              const std::vector<size_t>& layout_a = {},
              const std::vector<size_t>& layout_b = {},
              const std::vector<size_t>& layout_c = {});
    BrgemmCPU() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t get_iter_count() const {
        return m_iter_count;
    }

private:
    size_t m_iter_count;
};
}  // namespace ov::intel_cpu
