// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "transformations/snippets/x64/pass/lowered/brgemm_copy_b_loop_ports_adjuster.hpp"
#    include "transformations/snippets/x64/pass/lowered/external_repacking_adjuster.hpp"
#endif

namespace ov::intel_cpu {
using namespace ov::snippets::lowered::pass;

const size_t CPURuntimeConfigurator::rank6D = 6;

#ifdef SNIPPETS_DEBUG_CAPS
std::string CPURuntimeConfig::to_string() const {
    std::stringstream out;
    out << RuntimeConfig::to_string();
    out << "Loop Parameters:"
        << "\n";
    for (size_t i = 0; i < loop_args.size(); ++i) {
        const auto& loop = loop_args[i];
        out << "\t[" << i << "] WA: " << loop.m_work_amount << "\n";
        out << "\tPointer Increments: ";
        for (int64_t j = 0; j < loop.m_num_data_ptrs; ++j) {
            out << loop.m_ptr_increments[j] << " ";
        }
        out << "\n";
        out << "\tFinalization offsets: ";
        for (int64_t j = 0; j < loop.m_num_data_ptrs; ++j) {
            out << loop.m_finalization_offsets[j] << " ";
        }
        out << "\n";
    }
    return out.str();
}
#endif

CPURuntimeConfigurator::CPURuntimeConfigurator(ov::intel_cpu::MultiCacheWeakPtr cache)
    : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()),
      compiled_kernel_cache(std::move(cache)) {}

void CPURuntimeConfigurator::initialization(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    RuntimeConfigurator::initialization(linear_ir);
#ifdef OPENVINO_ARCH_X86_64
    RuntimeOptimizer::register_if_applicable<BrgemmCopyBLoopPortsAdjuster>(m_intermediate_optimizers, linear_ir, this);
    RuntimeOptimizer::register_if_applicable<BrgemmExternalRepackingAdjuster>(m_final_optimizers, linear_ir, this);
#endif
}

void CPURuntimeConfigurator::update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    RuntimeConfigurator::update(linear_ir);
    if (linear_ir->is_dynamic()) {
        update_loop_args(linear_ir);
    }
}

void CPURuntimeConfigurator::update_tensor_rank(const ov::snippets::VectorDims& master_shape) const {
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
        loop_arg = jit_snippets_call_args::loop_args_t(loop_info->get_work_amount(),
                                                       loop_info->get_ptr_increments(),
                                                       loop_info->get_finalization_offsets());
        for (int64_t i = 0; i < loop_arg.m_num_data_ptrs; ++i) {
            loop_arg.m_ptr_increments[i] *= (increment * data_sizes[i]);
            loop_arg.m_finalization_offsets[i] *= data_sizes[i];
        }
    }
}
}  // namespace ov::intel_cpu
