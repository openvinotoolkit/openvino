// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"


namespace ov {
namespace intel_cpu {

CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {
    std::static_pointer_cast<CPURuntimeConfig>(m_config)->m_kernel_executor_table = m_kernel_executor_table;
}

void CPURuntimeConfigurator::update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    RuntimeConfigurator::update(linear_ir);
    // todo: remove this
//    for (const auto& p : linear_ir->get_parameters()) {
//        std::cerr << "[";
//        for (const auto& s : p->get_output_port_descriptor(0)->get_shape())
//            std::cerr << s << " ";
//        std::cerr << "]\n";
//    }

    if (linear_ir->is_dynamic()) {
        update_kernel_executors(linear_ir);
        update_loop_args(linear_ir);
    }
}

void CPURuntimeConfigurator::init_tensor_rank(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) const {
    m_config->tensor_rank = std::max(linear_ir->get_master_shape().size(), rank6D);
}

void CPURuntimeConfigurator::update_loop_args(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) const {
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

void CPURuntimeConfigurator::update_kernel_executors(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_config);
    OPENVINO_ASSERT(cpu_config, "CPURuntimeConfigurator expects CPURuntimeConfig");
    m_kernel_executor_table->update_kernel_executors(linear_ir);
    cpu_config->kernel_exec_table_state = m_kernel_executor_table->get_state();
}

} // namespace intel_cpu
} // namespace ov
