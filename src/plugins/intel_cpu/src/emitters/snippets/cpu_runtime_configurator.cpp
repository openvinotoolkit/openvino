// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "transformations/snippets/x64/op/brgemm_cpu.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/loop_manager.hpp"


namespace ov {
namespace intel_cpu {

CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {
}

void CPURuntimeConfigurator::update(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    RuntimeConfigurator::update(linear_ir);

    if (linear_ir->is_dynamic()) {
        const auto& loop_manager = linear_ir->get_loop_manager();
        update_loop_args(loop_manager);
        update_brgemms(loop_manager);
        get_kernel_executor_table()->update_state();
    }
}

void CPURuntimeConfigurator::initialization(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) {
    RuntimeConfigurator::initialization(linear_ir);

    for (const auto& expr : *linear_ir) {
        if (ov::is_type<ov::intel_cpu::BrgemmCPU>(expr->get_node())) {
            const auto& in0_desc = expr->get_input_port_descriptor(0);
            const auto& in1_desc = expr->get_input_port_descriptor(1);
            const auto& out_desc = expr->get_output_port_descriptor(0);

            const auto& in0_subtensor = in0_desc->get_subtensor();
            const auto& in1_subtensor = in1_desc->get_subtensor();
            const auto& out_subtensor = out_desc->get_subtensor();

            // TODO [146125]: At the moment only blocking by dynamic M is supported
            //                So we save Brgemm with only dynamic M
            //                If there are other dynamic dimensions, throw exception for now
            OPENVINO_ASSERT(!snippets::utils::is_dynamic_value(*in0_subtensor.crbegin()) &&
                            !snippets::utils::is_dynamic_value(*in1_subtensor.crbegin()) &&
                            !snippets::utils::is_dynamic_value(*(++in1_subtensor.crbegin())) &&
                            !snippets::utils::is_dynamic_value(*out_subtensor.crbegin()),
                            "CPURuntimeConfigurator supports only dynamic M in Brgemm subtensors");
            OPENVINO_ASSERT(*(++in0_subtensor.crbegin()) == *(++out_subtensor.crbegin()),
                            "Incorrect values in subtensors of BrgemmCPU");

            if (snippets::utils::is_dynamic_value(*(++in0_subtensor.crbegin())))
                m_dynamic_brgemms.insert(expr);
        }
    }
}

void CPURuntimeConfigurator::init_tensor_rank(const std::shared_ptr<ov::snippets::lowered::LinearIR>& linear_ir) const {
    m_config->tensor_rank = std::max(linear_ir->get_master_shape().size(), rank6D);
}

void CPURuntimeConfigurator::update_loop_args(const ov::snippets::lowered::LoopManagerPtr& loop_manager) const {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_config);
    OPENVINO_ASSERT(cpu_config, "CPURuntimeConfigurator expects CPURuntimeConfig");

    const auto& loop_map = loop_manager->get_map();
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

void CPURuntimeConfigurator::update_brgemms(const ov::snippets::lowered::LoopManagerPtr& loop_manager) const {
    for (const auto& brgemm_expr : m_dynamic_brgemms) {
        const auto& loop_ids = brgemm_expr->get_loop_ids();
        OPENVINO_ASSERT(!loop_ids.empty(), "Dynamic Brgemm must be in loops");
        // TODO [146125]: Loop by M is first one in `loop_ids`
        const auto& expanded_loop_info = loop_manager->get_loop_info<snippets::lowered::ExpandedLoopInfo>(loop_ids.front());
        const auto& block_size_m = expanded_loop_info->get_work_amount();

        brgemm_expr->get_input_port_descriptor(0)->set_subtensor_value(1, block_size_m);
        brgemm_expr->get_output_port_descriptor(0)->set_subtensor_value(1, block_size_m);
    }
}

} // namespace intel_cpu
} // namespace ov
