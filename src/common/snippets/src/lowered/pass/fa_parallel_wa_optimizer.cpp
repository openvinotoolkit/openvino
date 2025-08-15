// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/fa_parallel_wa_optimizer.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <unordered_set>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/parameter.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/pass/runtime_optimizer.hpp"
#include "snippets/op/fa.hpp"
#include "snippets/pass/split_dimension_m.hpp"
#include "snippets/runtime_configurator.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::lowered::pass {
using namespace ov::snippets::pass;

const size_t FAParallelWAOptimizer::m_dim_M_idx = 1;

FAParallelWAOptimizer::FAParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir,
                                             const RuntimeConfigurator* configurator)
    : lowered::pass::RuntimeOptimizer(configurator) {
    if (linear_ir->get_config().m_enable_domain_optimization || !linear_ir->is_dynamic()) {
        return;
    }

    auto is_fa = [](const lowered::ExpressionPtr& expr) {
        return ov::is_type<op::FA>(expr->get_node());
    };
    auto fa_it = std::find_if(linear_ir->begin(), linear_ir->end(), is_fa);
    if (fa_it == linear_ir->end()) {
        has_dynamic_fa = false;
        return;
    }
    has_dynamic_fa = true;
    m_concurrency = linear_ir->get_config().m_min_parallel_work_amount;

    std::unordered_set<lowered::ExpressionPtr> fa = {*fa_it};
    m_unsqueezed_params = find_unsqueezed_params(linear_ir, fa);

    OPENVINO_ASSERT(!m_unsqueezed_params.empty(), "unsqueezed_params mustn't be empty after initialization");

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

std::unordered_set<size_t> FAParallelWAOptimizer::find_unsqueezed_params(
    const lowered::LinearIRCPtr& linear_ir,
    const std::unordered_set<lowered::ExpressionPtr>& fas) {
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
    for (const auto& fa : fas) {
        const auto& fa_b_input = fa->get_input_port_connector(1)->get_source().get_expr();
        utils::visit_path(fa_b_input, visited, add_param, true);
        const auto& fa_c_input = fa->get_input_port_connector(2)->get_source().get_expr();
        utils::visit_path(fa_c_input, visited, add_param, true);
    }
    return unsqueezed_params;
}

bool FAParallelWAOptimizer::run(const lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::FAParallelWAOptimizer")
    auto is_fa = [](const lowered::ExpressionPtr& expr) {
        return ov::is_type<op::FA>(expr->get_node());
    };
    auto fa_it = std::find_if(linear_ir.begin(), linear_ir.end(), is_fa);
    if (fa_it == linear_ir.end()) {
        return false;
    }
    auto& rt_info = (*fa_it)->get_node()->get_rt_info();
    rt_info.erase("splitm_kernel_dim");
    const auto& config = m_configurator->get_config();
    size_t new_batch_dim = 0, new_kernel_dim = 0;
    if (!SplitDimensionM::split(config->master_shape, m_concurrency, new_batch_dim, new_kernel_dim)) {
        return false;
    }
    auto& master_shape = config->master_shape;
    *++master_shape.rbegin() = new_kernel_dim;
    master_shape.insert(master_shape.cbegin() + master_shape.size() - 2, new_batch_dim);
    m_configurator->update_tensor_rank(master_shape);

    for (size_t i = 0; i < m_configurator->get_io_num(); ++i) {
        config->io_shapes[i] =
            m_unsqueezed_params.count(i)
                ? SplitDimensionM::unsqueeze_m_dim(config->io_shapes[i], m_dim_M_idces[i])
                : SplitDimensionM::reshape_m_dim(config->io_shapes[i], m_dim_M_idces[i], new_batch_dim, new_kernel_dim);
    }
    config->io_layouts = m_optimized_layouts;

    rt_info["splitm_kernel_dim"] = new_kernel_dim;

    return true;
}

}  // namespace ov::snippets::lowered::pass