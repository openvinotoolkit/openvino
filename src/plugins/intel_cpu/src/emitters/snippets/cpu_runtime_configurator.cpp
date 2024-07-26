// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/utils/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"


namespace ov {
namespace intel_cpu {

CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {
}

void CPURuntimeConfigurator::update(const ov::snippets::lowered::LinearIRPtr& linear_ir) {
    if (linear_ir->is_dynamic()) {
        update_loop_info(linear_ir);
        update_loop_args(linear_ir);
        // Update KernelExecutor Table should be before `update_buffer_scratchpad_size`
        // because `ComputeAllocationSize` depends on subtensors which are updated in the table
        get_kernel_executor_table()->update_state(linear_ir);
        update_buffer_scratchpad_size(linear_ir);
    }

    m_config->master_shape = linear_ir->get_master_shape();

    update_data_offsets();
    update_latest_shapes();
}

void CPURuntimeConfigurator::init_tensor_rank(const ov::snippets::lowered::LinearIRPtr& linear_ir) const {
    m_config->tensor_rank = std::max(linear_ir->get_master_shape().size(), rank6D);
}

void CPURuntimeConfigurator::update_loop_args(const ov::snippets::lowered::LinearIRPtr& linear_ir) const {
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
