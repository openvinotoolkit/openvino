// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/mha_parallel_wa_optimizer.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/utils/loop_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
using namespace ov::snippets::pass;

const size_t MHAParallelWAOptimizer::m_dim_M_idx = 1;

MHAParallelWAOptimizer::MHAParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, const RuntimeConfigurator* configurator)
    : lowered::pass::RuntimeOptimizer(configurator) {
    if (linear_ir->get_config().m_enable_domain_optimization || !linear_ir->is_dynamic())
        return;

    const auto brgemms = find_applicable_brgemms(linear_ir);
    if (brgemms.empty())
        return;

    m_concurrency = linear_ir->get_config().m_min_parallel_work_amount;
    m_unsqueezed_params = find_unsqueezed_params(linear_ir, brgemms);
    OPENVINO_ASSERT(!m_unsqueezed_params.empty(), "unsqueezed_params mustn't be empty after initialization");
    m_loops_to_split = find_loops_to_split(linear_ir, m_unsqueezed_params);

    m_dim_M_idces.resize(configurator->get_io_num());
    m_optimized_layouts.resize(configurator->get_io_num());
    for (size_t i = 0; i < configurator->get_io_num(); ++i) {
        const auto& layout = configurator->get_io_descs()[i]->get_layout();
        const auto dim_idx = i < configurator->get_in_num() ? utils::get_input_dim_idx(layout, m_dim_M_idx)
                                                            : utils::get_output_dim_idx(layout, m_dim_M_idx);
        m_dim_M_idces[i] = dim_idx;
        const auto m_idx = i < configurator->get_in_num() ? dim_idx : layout.size() - 2;
        m_optimized_layouts[i] = SplitDimensionM::get_updated_order(layout, m_idx);
    }
}

bool MHAParallelWAOptimizer::run(const lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MHAParallelWAOptimizer")
    const auto& config = m_configurator->get_config();
    size_t new_batch_dim, new_kernel_dim;
    if (!SplitDimensionM::split(config->master_shape, m_concurrency, new_batch_dim, new_kernel_dim))
        return false;
    auto& master_shape = config->master_shape;
    *++master_shape.rbegin() = new_kernel_dim;
    master_shape.insert(master_shape.cbegin() + master_shape.size() - 2, new_batch_dim);
    m_configurator->update_tensor_rank(master_shape);

    RuntimeConfigurator::LoopInfoRuntimeParamsMap initialized_info;
    auto updater = [&](const lowered::LoopInfoPtr& loop_info) {
        if (const auto unified_loop_info = ov::as_type_ptr<lowered::UnifiedLoopInfo>(loop_info)) {
            if (initialized_info.count(unified_loop_info) == 0) {
                if (!ov::is_type<lowered::InnerSplittedUnifiedLoopInfo>(unified_loop_info))
                    unified_loop_info->set_work_amount(new_kernel_dim);
                snippets::utils::update_data_pointer_shifts(unified_loop_info);
                initialized_info[unified_loop_info] = RuntimeConfigurator::get_loop_runtime_params(unified_loop_info);
            }
        } else if (const auto expanded_loop_info = ov::as_type_ptr<lowered::ExpandedLoopInfo>(loop_info)) {
            m_configurator->update_expanded_loop_info(expanded_loop_info, initialized_info);
        } else {
            OPENVINO_THROW("Failed to update loop info: unknown type!");
        }
    };
    lowered::LoopInfoSet updated_loops;
    for (const auto& loop : m_loops_to_split) {
        loop->apply(updater, updated_loops);
    }

    for (size_t i = 0; i < m_configurator->get_io_num(); ++i) {
        config->io_shapes[i] = m_unsqueezed_params.count(i)
                        ? SplitDimensionM::unsqueeze_m_dim(config->io_shapes[i], m_dim_M_idces[i])
                        : SplitDimensionM::reshape_m_dim(config->io_shapes[i], m_dim_M_idces[i], new_batch_dim, new_kernel_dim);
    }
    config->io_layouts = m_optimized_layouts;
    return true;
}

std::unordered_set<lowered::ExpressionPtr> MHAParallelWAOptimizer::find_applicable_brgemms(
    const lowered::LinearIRCPtr& linear_ir,
    bool check_dynamic_wa) {
    auto is_brgemm = [](const lowered::ExpressionPtr& expr) {
        return ov::is_type<op::Brgemm>(expr->get_node());
    };
    auto brgemm_it = std::find_if(linear_ir->begin(), linear_ir->end(), is_brgemm);
    std::unordered_set<lowered::ExpressionPtr> brgemms;
    while (brgemm_it != linear_ir->end()) {
        brgemms.insert(*brgemm_it);
        brgemm_it = std::find_if(std::next(brgemm_it), linear_ir->end(), is_brgemm);
    }
    const auto& loop_manager = linear_ir->get_loop_manager();
    auto applicable_brgemm = [&loop_manager, check_dynamic_wa](const lowered::ExpressionPtr& expr) {
        const auto& loop_idces = expr->get_loop_ids();
        if (loop_idces.empty())
            return false;
        const auto& outermost_loop = loop_manager->get_loop_info(loop_idces[0]);
        if (check_dynamic_wa && !snippets::utils::is_dynamic_value(outermost_loop->get_work_amount()))
            return false;
        bool loop_by_m = true;
        outermost_loop->iterate_through_ports([&loop_by_m](const lowered::LoopPort& port) {
            if (port.is_processed() && port.get_dim_idx() != m_dim_M_idx)
                loop_by_m = false;
        });
        return loop_by_m;
    };
    return std::all_of(brgemms.begin(), brgemms.end(), applicable_brgemm) ? brgemms : std::unordered_set<lowered::ExpressionPtr>{};
}

std::unordered_set<size_t> MHAParallelWAOptimizer::find_unsqueezed_params(
    const lowered::LinearIRCPtr& linear_ir,
    const std::unordered_set<lowered::ExpressionPtr>& brgemms) {
    const auto& params = linear_ir->get_parameters();
    std::unordered_set<size_t> unsqueezed_params;
    auto add_param = [&params, &unsqueezed_params](const lowered::ExpressionPtr& expr) {
        if (ov::is_type<ov::op::v0::Parameter>(expr->get_node())) {
            auto found_param = std::find(params.begin(), params.end(), expr);
            OPENVINO_ASSERT(found_param != params.end(), "find_param didn't found parameter for expr");
            unsqueezed_params.insert(std::distance(params.begin(), found_param));
        }
    };

    std::unordered_set<lowered::ExpressionPtr> visited;
    for (const auto& brgemm : brgemms) {
        const auto& brgemm_b_input = brgemm->get_input_port_connector(1)->get_source().get_expr();
        utils::visit_path(brgemm_b_input, visited, add_param, true);
    }
    return unsqueezed_params;
}

std::vector<lowered::ExpandedLoopInfoPtr> MHAParallelWAOptimizer::find_loops_to_split(
    const lowered::LinearIRCPtr& linear_ir,
    const std::unordered_set<size_t>& unsqueezed_params) {
    const auto loop_manager = linear_ir->get_loop_manager();
    std::set<size_t> loop_idces_to_split;
    std::vector<size_t> prev_loop_idces;

    auto add_loop_idx_to_split = [&](const lowered::ExpressionPtr& expr) {
        const auto& loop_idces = expr->get_loop_ids();
        if (loop_idces != prev_loop_idces) {
            prev_loop_idces = loop_idces;
            for (const auto& loop_id : loop_idces) {
                const auto expanded_loop_info = loop_manager->get_loop_info<lowered::ExpandedLoopInfo>(loop_id);
                if (expanded_loop_info->get_dim_idx() == m_dim_M_idx) {
                    loop_idces_to_split.insert(loop_id);
                }
            }
        }
    };

    size_t i = 0;
    std::unordered_set<lowered::ExpressionPtr> visited;
    for (const auto& param : linear_ir->get_parameters()) {
        if (unsqueezed_params.count(i++))
            continue;
        utils::visit_path(param, visited, add_loop_idx_to_split, false);
    }

    const auto& loops_map = linear_ir->get_loop_manager()->get_map();
    std::vector<lowered::ExpandedLoopInfoPtr> loops_to_split;
    for (const auto& id : loop_idces_to_split)
        loops_to_split.push_back(ov::as_type_ptr<lowered::ExpandedLoopInfo>(loops_map.at(id)));
    return loops_to_split;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov