// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/mha_parallel_wa_optimizer.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/utils/loop_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
using namespace ov::snippets::pass;
namespace {
std::vector<size_t> get_updated_order(const std::vector<size_t>& order, size_t m_index) {
    std::vector<size_t> new_order(order.size() + 1, 0);
    size_t shift_idx = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] < m_index) {
            new_order[i + shift_idx] = order[i];
        } else if (order[i] == m_index) {
            new_order[i + shift_idx++] = order[i];
            new_order[i + shift_idx] = order[i] + 1;
        } else {
            new_order[i + shift_idx] = order[i] + 1;
        }
    }
    return new_order;
}

ov::snippets::VectorDims unsqueeze_m_dim(ov::snippets::VectorDims shape, size_t m_index) {
    shape.insert(shape.begin() + m_index, 1);
    return shape;
}

ov::snippets::VectorDims reshape_m_dim(ov::snippets::VectorDims shape, size_t m_index, size_t batch_m_dim, size_t new_m_dim) {
    if (shape[m_index] == 1)
        return unsqueeze_m_dim(std::move(shape), m_index);
    shape[m_index] = new_m_dim;
    shape.insert(shape.begin() + m_index, batch_m_dim);
    return shape;
}

bool is_prime_number(size_t value) {
    if (ov::snippets::utils::one_of(value, 2lu, 3lu))
        return true;
    if (value == 1 || value % 2 == 0 || value % 3 == 0)
        return false;
    const auto root = std::sqrt(value) + 1;
    for (size_t divisor = 5; divisor < root; divisor += 6) {
        if ((value % divisor == 0) || (value % (divisor + 2) == 0))
            return false;
    }
    return true;
}
} // namespace

const size_t MHAParallelWAOptimizer::m_dim_M_idx = 1;
const size_t MHAParallelWAOptimizer::m_min_kernel_m = 32;

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
    std::cout << "[ INFO ] m_loops_to_split size = " << m_loops_to_split.size() << std::endl;

    m_dim_M_idces.resize(configurator->get_io_num());
    m_optimized_layouts.resize(configurator->get_io_num());
    for (size_t i = 0; i < configurator->get_io_num(); ++i) {
        const auto& layout = configurator->get_io_descs()[i]->get_layout();
        const auto dim_idx = i < configurator->get_in_num() ? utils::get_input_dim_idx(layout, m_dim_M_idx)
                                                            : utils::get_output_dim_idx(layout, m_dim_M_idx);
        m_dim_M_idces[i] = dim_idx;
        const auto m_idx = i < configurator->get_in_num() ? dim_idx : layout.size() - 2;
        m_optimized_layouts[i] = get_updated_order(layout, m_idx);
    }
}

bool MHAParallelWAOptimizer::run(const lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::MHAParallelWAOptimizer")
    const auto& config = m_configurator->get_config();
    size_t new_batch_dim, new_kernel_dim;
    bool split_res = split(config->master_shape, m_concurrency, new_batch_dim, new_kernel_dim);
    std::cout << "Split result: " << split_res << ", Master shape: " << ov::PartialShape(config->master_shape)
              << ", Concurrency: " << m_concurrency << ", New batch dimension: " << new_batch_dim
              << ", New kernel dimension: " << new_kernel_dim << std::endl;

    if (!split_res)
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
                        ? unsqueeze_m_dim(config->io_shapes[i], m_dim_M_idces[i])
                        : reshape_m_dim(config->io_shapes[i], m_dim_M_idces[i], new_batch_dim, new_kernel_dim);
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
        const auto brgemm_b_input = brgemm->get_input_expr_ptr(1);
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

bool MHAParallelWAOptimizer::split(const ov::Shape& shape,
                                   size_t optimal_parallelism_work_amount,
                                   size_t& batch_m_dim,
                                   size_t& new_m_dim) {
    const auto batch_dim = std::accumulate(shape.rbegin() + 2, shape.rend(), size_t(1), std::multiplies<size_t>());
    const auto m_dim = *(shape.rbegin() + 1);
    if (is_prime_number(m_dim))
        return false;

    // We skip optimization if the current batch is optimal for concurrency
    if (batch_dim % optimal_parallelism_work_amount == 0)
        return false;

    auto split_is_done = [&batch_m_dim]() {
        return batch_m_dim != 1;
    };

    std::tie(batch_m_dim, new_m_dim) = split_ideally(batch_dim, m_dim, optimal_parallelism_work_amount);
    if (split_is_done())
        return true;

    std::tie(batch_m_dim, new_m_dim) = split_minimize_kernel_wa(batch_dim, m_dim, optimal_parallelism_work_amount);
    if (split_is_done())
        return true;
    // If all the previous heuristics failed, fallback heuristic is used, which reflects the old splitting behavior
    if (batch_dim < optimal_parallelism_work_amount)
        std::tie(batch_m_dim, new_m_dim) =
            split_fallback_increase_parallel_wa(batch_dim, m_dim, optimal_parallelism_work_amount);
    return split_is_done();
}

std::pair<size_t, size_t> MHAParallelWAOptimizer::split_ideally(size_t batch_dim,
                                                                size_t m_dim,
                                                                size_t optimal_parallelism_work_amount) {
    // Ideal case #1: M can be split on the parts one of which complements the batch dimension to the optimal parallel
    // work amount In this case, each thread will execute the Snippets kernel once
    const size_t lower_bound = optimal_parallelism_work_amount / batch_dim;
    if (lower_bound * batch_dim == optimal_parallelism_work_amount && m_dim % lower_bound == 0)
        return std::make_pair(lower_bound, m_dim / lower_bound);

    // Ideal case #2: M is divisible by optimal parallel work amount, and the new_m_dim is big enough
    // In this case, each thread will execute the Snippets kernel 'batch_dim' times
    if (m_dim % optimal_parallelism_work_amount == 0) {
        const auto new_m_dim = m_dim / optimal_parallelism_work_amount;
        if (new_m_dim >= m_min_kernel_m)
            return std::make_pair(optimal_parallelism_work_amount, new_m_dim);
    }

    return std::make_pair(1, m_dim);
}

std::pair<size_t, size_t> MHAParallelWAOptimizer::split_fallback_increase_parallel_wa(
    size_t batch_dim,
    size_t m_dim,
    size_t optimal_parallelism_work_amount) {
    std::pair<size_t, size_t> splited = {1, m_dim};
    const size_t upper_bound = utils::div_up(2 * optimal_parallelism_work_amount, batch_dim);
    for (size_t divisor_0 = upper_bound - 1; divisor_0 > 1; divisor_0--) {
        size_t divisor_1 = m_dim / divisor_0;
        if (divisor_1 * divisor_0 == m_dim)
            return divisor_0 * batch_dim >= optimal_parallelism_work_amount ? std::make_pair(divisor_0, divisor_1)
                                                                            : splited;
    }
    return splited;
}

std::pair<size_t, size_t> MHAParallelWAOptimizer::split_minimize_kernel_wa(size_t batch_dim,
                                                                           size_t m_dim,
                                                                           size_t optimal_parallelism_work_amount) {
    // This heuristic minimizes 'm_kernel' (=> maximizes 'm_batch') with a limitation that 'm_kernel >= min_kernel_m'.
    // In other words, it tries to find 'm_kernel' bigger than 'm_min_kernel_m' and at the same time as close as possible
    // to this value.
    std::pair<size_t, size_t> best_result = {1, m_dim};
    for (size_t divisor = 2; divisor < std::sqrt(m_dim); ++divisor) {
        if (m_dim % divisor != 0)
            continue;
        // If divisor is more than 'm_min_kernel_m', divisor becomes 'm_kernel',
        // guaranteeing the most optimal implementation from 'm_kernel' minimization perspective.
        if (divisor >= m_min_kernel_m)
            return std::make_pair(m_dim / divisor, divisor);

        // If divisor is less than 'm_min_kernel_m', divisor becomes m_batch.
        // However, it is not guaranteed that the current 'm_kernel = m_dim / divisor' is minimized, as one of the next
        // divisors can be more optimal. So in this case the best result is remembered
        const size_t m_kernel = m_dim / divisor;
        if (m_kernel >= m_min_kernel_m) {
            best_result.first = divisor;
            best_result.second = m_kernel;
        }
    }
    if (best_result.first * batch_dim >= optimal_parallelism_work_amount)
        return best_result;
    return std::make_pair(1, m_dim);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov