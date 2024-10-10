// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

const size_t CPURuntimeConfigurator::rank6D = 6;

#ifdef SNIPPETS_DEBUG_CAPS
std::string CPURuntimeConfig::to_string() const {
    std::stringstream out;
    out << RuntimeConfig::to_string();
    out << "Loop Parameters:" << "\n";
    for (size_t i = 0; i < loop_args.size(); ++i) {
        const auto& loop = loop_args[i];
        out << "\t[" << i << "] WA: " << loop.m_work_amount << "\n";
        out << "\tPointer Increments: ";
        for (int64_t j = 0; j < loop.m_num_data_ptrs; ++j)
            out << loop.m_ptr_increments[j] << " ";
        out << "\n";
        out << "\tFinalization offsets: ";
        for (int64_t j = 0; j < loop.m_num_data_ptrs; ++j)
            out << loop.m_finalization_offsets[j] << " ";
        out << "\n";
    }
    return out.str();
}
#endif

CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {
}

void CPURuntimeConfigurator::update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_config);
    auto& optimal_descs = cpu_config->m_in_requested_descs;
    optimal_descs.resize(m_in_num);
    const auto& params = linear_ir->get_parameters();
    OPENVINO_ASSERT(params.size() == m_in_num);
    for (size_t i = 0; i < m_in_num; ++i) {
        // TODO: remove
        if (i != 1) continue;
        const auto& param = params[i];
        const auto& shape = param->get_output_port_descriptor(0)->get_shape();
        VectorDims normalized_dims(3, 1);
        *normalized_dims.rbegin() = *shape.rbegin();
        *++normalized_dims.rbegin() = *++shape.rbegin();
        normalized_dims[0] = std::accumulate(shape.begin(), shape.end() - 2, static_cast<Dim>(1), std::multiplies<Dim>());
        optimal_descs[i] = std::make_shared<DnnlBlockedMemoryDesc>(Shape(normalized_dims),
                                                                   dnnl::memory::data_type::bf16,
                                                                   dnnl::memory::format_tag::aCB16b64c2b);
    }

    RuntimeConfigurator::update(linear_ir);
    if (linear_ir->is_dynamic())
        update_loop_args(linear_ir);

    for (size_t i = 0; i < m_in_num; ++i) {
        if (optimal_descs[i]) {
            auto& offsets = m_config->io_data_offsets[i];
            // TODO: remove this hardcode
            if (!std::getenv("REFERENCE") && i == 1)
                offsets[3] = 2048 * 2;
        }
    }
}

void CPURuntimeConfigurator::update_tensor_rank(const ov::snippets::VectorDims& master_shape) {
    m_config->tensor_rank = std::max(master_shape.size(), rank6D);
}

void CPURuntimeConfigurator::init_tensor_rank(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const {
    m_config->tensor_rank = std::max(linear_ir->get_master_shape().size(), rank6D);
}

void CPURuntimeConfigurator::update_loop_args(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_config);
    OPENVINO_ASSERT(cpu_config, "CPURuntimeConfigurator expects CPURuntimeConfig");

    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    cpu_config->loop_args.resize(loop_map.size());
    for (const auto& loop : loop_map) {
        const auto& idx = loop.first;
        const auto& loop_info = ov::as_type_ptr<ov::snippets::lowered::ExpandedLoopInfo>(loop.second);
        OPENVINO_ASSERT(loop_info, "CPURuntimeConfigurator expects ExpandedLoopInfo in loop manager");

        const auto& increment = loop_info->get_increment();
        const auto& data_sizes = loop_info->get_data_sizes();

        auto& loop_arg = cpu_config->loop_args[idx];
        loop_arg = jit_snippets_call_args::loop_args_t(loop_info->get_work_amount(), loop_info->get_ptr_increments(), loop_info->get_finalization_offsets());
        for (int64_t i = 0; i < loop_arg.m_num_data_ptrs; ++i) {
            loop_arg.m_ptr_increments[i] *= (increment * data_sizes[i]);
            loop_arg.m_finalization_offsets[i] *= data_sizes[i];
        }
    }
}

} // namespace intel_cpu
} // namespace ov
