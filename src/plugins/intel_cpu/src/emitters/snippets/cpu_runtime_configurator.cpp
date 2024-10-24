// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"

#ifndef OPENVINO_ARCH_ARM64
#include "transformations/snippets/x64/pass/lowered/adjust_brgemm_copy_b_loop_ports.hpp"
#endif
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

void CPURuntimeConfigurator::initialization(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    RuntimeConfigurator::initialization(linear_ir);
    if (linear_ir->is_dynamic()) {
        loopPortsAdjuster = BrgemmCopyBLoopPortsAdjuster(linear_ir, this);
    }
}

void CPURuntimeConfigurator::update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    m_config->master_shape = linear_ir->get_master_shape();
    if (linear_ir->is_dynamic()) {
        update_loop_info(linear_ir);
    }

    if (!m_optimizer.optimize()) {
        // If the optimization was not applied, offsets are updated using shapes from descriptors
        auto shapes = extract_shapes();
        update_data_offsets(shapes, extract_layouts());
        m_latest_shapes = std::move(shapes);
    }
    if (linear_ir->is_dynamic())
        loopPortsAdjuster.optimize();

    // Update KernelExecutor Table should be before `update_buffer_scratchpad_size`
    // because `ComputeAllocationSize` depends on subtensors which are updated in the table
    get_kernel_executor_table()->update_state(linear_ir);
    update_buffer_scratchpad_size(linear_ir);

    if (linear_ir->is_dynamic()) {
        update_loop_args(linear_ir);
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
#ifdef OPENVINO_ARCH_ARM64
CPURuntimeConfigurator::BrgemmCopyBLoopPortsAdjuster::BrgemmCopyBLoopPortsAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                                                   ov::intel_cpu::CPURuntimeConfigurator *configurator) {
}

void CPURuntimeConfigurator::BrgemmCopyBLoopPortsAdjuster::optimize() {
}
#else
CPURuntimeConfigurator::BrgemmCopyBLoopPortsAdjuster::BrgemmCopyBLoopPortsAdjuster(const ov::snippets::lowered::LinearIRCPtr& linear_ir,
                                                                                   ov::intel_cpu::CPURuntimeConfigurator *configurator) {
    const auto& pass = std::make_shared<intel_cpu::pass::AdjustBrgemmCopyBLoopPorts>();
    pass->run(*linear_ir);
    const auto& affected_uni_loops = pass->get_affected_loops();
    const auto& loop_map = linear_ir->get_loop_manager()->get_map();
    for (const auto& p : loop_map) {
        if (const auto& exp_loop = ov::as_type_ptr<snippets::lowered::ExpandedLoopInfo>(p.second)) {
            const auto& uni_loop = exp_loop->get_unified_loop_info();
            if (affected_uni_loops.count(uni_loop))
                m_affected_uni2exp_map[uni_loop].push_back(exp_loop);
        }
    }
}

void CPURuntimeConfigurator::BrgemmCopyBLoopPortsAdjuster::optimize() {
        for (const auto& p : m_affected_uni2exp_map) {
            const auto& uni_loop = p.first;
            const auto& exp_loops = p.second;
            snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap initialized_info;
            if (intel_cpu::pass::AdjustBrgemmCopyBLoopPorts::update_loop_info(uni_loop)) {
                initialized_info[uni_loop] = get_loop_runtime_params(uni_loop);
                for (const auto& exp_loop : exp_loops)
                    update_expanded_loop_info(exp_loop, initialized_info);
            }
        }
}
#endif

} // namespace intel_cpu
} // namespace ov
