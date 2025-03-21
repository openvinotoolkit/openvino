// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "brgemm_cpu.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {
using namespace brgemm_utils;

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     size_t iter_count,
                     BRGEMM_TYPE type,
                     const size_t offset_a,
                     const size_t offset_b,
                     const size_t offset_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : GemmCPU(A, B, type, offset_a, offset_b, offset_c, layout_a, layout_b, layout_c),
      m_iter_count(iter_count) {}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     const Output<Node>& scratch,
                     size_t iter_count,
                     BRGEMM_TYPE type,
                     const size_t offset_a,
                     const size_t offset_b,
                     const size_t offset_scratch,
                     const size_t offset_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : GemmCPU(A, B, scratch, type, offset_a, offset_b, offset_scratch, offset_c, layout_a, layout_b, layout_c),
      m_iter_count(iter_count) {}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     size_t iter_count,
                     BRGEMM_TYPE type,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : GemmCPU(A, B, type, desc_a, desc_b, desc_c, layout_a, layout_b, layout_c),
      m_iter_count(iter_count) {}

BrgemmCPU::BrgemmCPU(const Output<Node>& A,
                     const Output<Node>& B,
                     const Output<Node>& scratch,
                     size_t iter_count,
                     BRGEMM_TYPE type,
                     const PortDescriptor& desc_a,
                     const PortDescriptor& desc_b,
                     const PortDescriptor& desc_scratch,
                     const PortDescriptor& desc_c,
                     const std::vector<size_t>& layout_a,
                     const std::vector<size_t>& layout_b,
                     const std::vector<size_t>& layout_c)
    : GemmCPU(A, B, scratch, type, desc_a, desc_b, desc_scratch, desc_c, layout_a, layout_b, layout_c),
      m_iter_count(iter_count) {}

std::shared_ptr<Node> BrgemmCPU::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(BrgemmCPU_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    std::shared_ptr<BrgemmCPU> brgemm;
    if (!with_scratchpad(m_type)) {
        return std::make_shared<BrgemmCPU>(
            new_args.at(0),
            new_args.at(1),
            m_iter_count,
            m_type,
            get_input_port_descriptor(0),
            get_input_port_descriptor(1),
            get_output_port_descriptor(0),
            snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
            snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
            snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
    }
    return std::make_shared<BrgemmCPU>(
        new_args.at(0),
        new_args.at(1),
        new_args.at(2),
        m_iter_count,
        m_type,
        get_input_port_descriptor(0),
        get_input_port_descriptor(1),
        get_input_port_descriptor(2),
        get_output_port_descriptor(0),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(0))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(input(1))->get_layout(),
        snippets::lowered::PortDescriptorUtils::get_port_descriptor_ptr(output(0))->get_layout());
}

}  // namespace ov::intel_cpu
