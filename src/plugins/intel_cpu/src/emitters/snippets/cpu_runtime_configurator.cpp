// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/utils.hpp"
#include "transformations/snippets/x64/op/brgemm_cpu.hpp"

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
    // TODO: rename
    out << "External indices:"
        << "\n";
    for (const auto& idx : external_ptrs_idces) {
        out << idx << " ";
    }
    out << "\n";
    return out.str();
}
#endif

CPURuntimeConfigurator::CPURuntimeConfigurator(ov::intel_cpu::MultiCacheWeakPtr cache)
    : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()),
      compiled_kernel_cache(std::move(cache)) {}

void CPURuntimeConfigurator::initialization(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    RuntimeConfigurator::initialization(linear_ir);
    init_external_ptrs(linear_ir);
#ifdef OPENVINO_ARCH_X86_64
    using namespace pass;
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

void CPURuntimeConfigurator::init_external_ptrs(const ov::snippets::lowered::LinearIRCPtr& linear_ir) const {
    const auto& cpu_config = ov::as_type_ptr<CPURuntimeConfig>(m_config);
    OPENVINO_ASSERT(cpu_config, "CPURuntimeConfigurator expects CPURuntimeConfig");

    const auto& parameters = linear_ir->get_parameters();
    size_t external_ptrs_count = 0;
    for (size_t i = 0; i < parameters.size(); ++i) {
        const auto& param_expr = parameters[i];
        const auto& param_info = param_expr->get_node()->get_rt_info();
        if (param_info.count("POSTOP_INPUT")) {
            cpu_config->external_ptrs_idces.insert(i);
            std::cout << "[ INFO ] CPURuntimeConfigurator::init_external_ptrs - POSTOP_INPUT: " << i << std::endl;
            for (const auto& connector : param_expr->get_output_port_connectors()) {
                for (const auto& consumer : connector->get_consumers()) {
                    const auto consumer_node = consumer.get_expr()->get_node();
                    auto& rt_info = consumer_node->get_rt_info();
                    // TODO: this communication must be done in a more transparent way then using RT info
                    if (ov::is_type<ov::intel_cpu::BrgemmCPU>(consumer_node) && !rt_info.count("EXTERNAL_PTR_OFFSET")) {
                        rt_info["EXTERNAL_PTR_OFFSET"] = external_ptrs_count;
                    }
                }
            }
            external_ptrs_count++;
        }
    }
}
}  // namespace ov::intel_cpu
