// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "emitters/snippets/cpu_runtime_configurator.hpp"

#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

using namespace ov::snippets::lowered;
using namespace ov::snippets::pass;

const size_t CPURuntimeConfigurator::rank6D = 6;

CPURuntimeConfigurator::CPURuntimeConfigurator() : ov::snippets::RuntimeConfigurator(std::make_shared<CPURuntimeConfig>()) {
}

void CPURuntimeConfigurator::update(const ov::snippets::lowered::LinearIRCPtr& linear_ir) {
    m_config->master_shape = linear_ir->get_master_shape();

    ov::snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap initialized_info;
    std::vector<ov::snippets::VectorDims> updated_shapes;
    std::vector<std::vector<size_t>> updated_layouts;
    bool optimize_work_amount = m_optimizer.need_optimize(m_config->master_shape);
    if (optimize_work_amount) {
        m_optimizer.update_split_loops_info(initialized_info);
        m_optimizer.update_shapes(m_io_descs, updated_shapes, m_in_num);
        m_optimizer.update_layouts(m_io_descs, updated_layouts, m_in_num);
        m_optimizer.update_config(m_config);
    }

    if (linear_ir->is_dynamic()) {
        update_loop_info(linear_ir, initialized_info);
        update_loop_args(linear_ir);
        // Update KernelExecutor Table should be before `update_buffer_scratchpad_size`
        // because `ComputeAllocationSize` depends on subtensors which are updated in the table
        get_kernel_executor_table()->update_state(linear_ir);
        update_buffer_scratchpad_size(linear_ir);
    }

    update_data_offsets(updated_shapes, updated_layouts);
    // TODO: unify this logic somehow?
    if (!optimize_work_amount) {
        update_latest_shapes();
    } else {
        m_latest_shapes = std::move(updated_shapes);
    }
}

void CPURuntimeConfigurator::initialization(const ov::snippets::lowered::LinearIRPtr& linear_ir) {
    RuntimeConfigurator::initialization(linear_ir);
    m_optimizer.init(linear_ir);
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

void CPURuntimeConfigurator::ParallelWAOptimizer::init(const ov::snippets::lowered::LinearIRPtr& linear_ir) {
    if (linear_ir->get_config().m_enable_domain_optimization || !linear_ir->is_dynamic())
        return;
    concurrency = linear_ir->get_config().m_min_parallel_work_amount;

    const auto brgemms = find_applicable_brgemms(linear_ir);
    // Parallel WA optimization is Brgemm related
    if (brgemms.empty())
        return;

    init_non_m_related_params(linear_ir, brgemms);
    OPENVINO_ASSERT(!not_m_related_params.empty(), "not_m_related_params mustn't be empty after initialization");
    init_loops_to_split(linear_ir);
}

bool CPURuntimeConfigurator::ParallelWAOptimizer::need_optimize(const ov::snippets::VectorDims& master_shape) {
    return !loops_to_split.empty() && SplitDimensionM::split(master_shape, concurrency, batch_m, new_m);
}

void CPURuntimeConfigurator::ParallelWAOptimizer::update_split_loops_info(
    ov::snippets::RuntimeConfigurator::LoopInfoRuntimeParamsMap& initialized_info) {
    for (const auto& loop : loops_to_split) {
        if (initialized_info.count(loop) == 0) {
            loop->set_work_amount(new_m);
            ov::snippets::lowered::pass::InitLoops::update_runtime_parameters(loop, false);
            initialized_info[loop] = compute_runtime_params(loop);
        }
    }
}

void CPURuntimeConfigurator::ParallelWAOptimizer::update_shapes(
    const std::vector<snippets::lowered::PortDescriptorPtr>& io_descs,
    std::vector<ov::snippets::VectorDims>& shapes,
    size_t in_num) {
    shapes.resize(io_descs.size());
    for (size_t i = 0; i < io_descs.size(); ++i) {
        const auto& desc = io_descs[i];
        const auto dim_idx = i < in_num ? ov::snippets::utils::get_input_dim_idx(desc->get_layout(), 1)
                                        : ov::snippets::utils::get_output_dim_idx(desc->get_layout(), 1);
        shapes[i] = not_m_related_params.count(i)
                        ? SplitDimensionM::unsqueeze_m_dim(desc->get_shape(), dim_idx)
                        : SplitDimensionM::reshape_m_dim(desc->get_shape(), dim_idx, batch_m, new_m);
    }
}

void CPURuntimeConfigurator::ParallelWAOptimizer::update_layouts(
    const std::vector<snippets::lowered::PortDescriptorPtr>& io_descs,
    std::vector<std::vector<size_t>>& layouts,
    size_t in_num) {
    layouts.resize(io_descs.size());
    for (size_t i = 0; i < io_descs.size(); ++i) {
        const auto& original_layout = io_descs[i]->get_layout();
        const auto dim_idx =
            i < in_num ? ov::snippets::utils::get_input_dim_idx(original_layout, 1) : original_layout.size() - 2;
        layouts[i] = SplitDimensionM::get_updated_order(original_layout, dim_idx);
    }
}

void CPURuntimeConfigurator::ParallelWAOptimizer::update_config(const std::shared_ptr<ov::snippets::RuntimeConfig>& config) {
    *++config->master_shape.rbegin() = new_m;
    config->master_shape.insert(config->master_shape.cbegin() + config->master_shape.size() - 2, batch_m);
    config->tensor_rank = std::max(config->master_shape.size(), CPURuntimeConfigurator::rank6D);
}

std::unordered_set<snippets::lowered::ExpressionPtr> CPURuntimeConfigurator::ParallelWAOptimizer::find_applicable_brgemms(
    const ov::snippets::lowered::LinearIRPtr& linear_ir) {
    auto is_brgemm = [](const ExpressionPtr& expr) {
        return ov::is_type<ov::snippets::op::Brgemm>(expr->get_node());
    };
    auto brgemm_it = std::find_if(linear_ir->begin(), linear_ir->end(), is_brgemm);
    std::unordered_set<ExpressionPtr> brgemms;
    while (brgemm_it != linear_ir->end()) {
        brgemms.insert(*brgemm_it);
        brgemm_it = std::find_if(std::next(brgemm_it), linear_ir->end(), is_brgemm);
    }
    const auto& loop_manager = linear_ir->get_loop_manager();
    auto applicable_brgemm = [&loop_manager](const ExpressionPtr& expr) {
        const auto& loop_idces = expr->get_loop_ids();
        if (loop_idces.empty())
            return false;
        const auto& outermost_loop = loop_manager->get_loop_info(loop_idces[0]);
        // WA: the feature doesn't work in static case
        if (!ov::snippets::utils::is_dynamic_value(outermost_loop->get_work_amount()))
            return false;
        bool loop_by_m = true;
        outermost_loop->iterate_through_ports([&loop_by_m](const LoopPort& port) {
            if (port.is_incremented && port.dim_idx != 1)
                loop_by_m = false;
        });
        return loop_by_m;
    };
    if (std::all_of(brgemms.begin(), brgemms.end(), applicable_brgemm))
        return brgemms;
    else
        return {};
}

void CPURuntimeConfigurator::ParallelWAOptimizer::init_non_m_related_params(
    const ov::snippets::lowered::LinearIRPtr& linear_ir,
    const std::unordered_set<snippets::lowered::ExpressionPtr>& brgemms) {
    const auto& params = linear_ir->get_parameters();
    auto add_param = [&params, this](const ExpressionPtr& expr) {
        if (ov::is_type<ov::op::v0::Parameter>(expr->get_node())) {
            auto found_param = std::find(params.begin(), params.end(), expr);
            OPENVINO_ASSERT(found_param != params.end(), "find_param didn't found parameter for expr");
            not_m_related_params.insert(std::distance(params.begin(), found_param));
        }
    };

    std::unordered_set<ExpressionPtr> visited;
    for (const auto& brgemm : brgemms) {
        const auto& brgemm_b_input = brgemm->get_input_port_connector(1)->get_source().get_expr();
        ov::snippets::utils::visit_path(brgemm_b_input, visited, add_param, true);
    }
}

void CPURuntimeConfigurator::ParallelWAOptimizer::init_loops_to_split(
    const ov::snippets::lowered::LinearIRPtr& linear_ir) {
    const auto loop_manager = linear_ir->get_loop_manager();

    auto add_loops_to_split = [&loop_manager, this](const ExpressionPtr& expr) {
        const auto& loop_ids = expr->get_loop_ids();
        if (!loop_ids.empty()) {
            const auto outermost_loop_idx = loop_ids[0];
            const auto loop_info_to_add = loop_manager->get_loop_info<ExpandedLoopInfo>(outermost_loop_idx);
            loops_to_split.insert(loop_info_to_add->get_unified_loop_info());
        }
    };

    size_t i = 0;
    std::unordered_set<ExpressionPtr> visited;
    // The idea is to traverse LIR down from the M dimension related parameters
    // and find all the outermost loops: these loops will be split in runtime
    for (const auto& param : linear_ir->get_parameters()) {
        // Ops after non related params mustn't be traversed
        if (not_m_related_params.count(i++))
            continue;
        ov::snippets::utils::visit_path(param, visited, add_loops_to_split, false);
    }
}

} // namespace intel_cpu
} // namespace ov
