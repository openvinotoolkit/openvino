// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/parallel_wa_optimizer.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/mha_parallel_wa_optimizer.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/utils/loop_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
using namespace ov::snippets::lowered;

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
} // namespace

ParallelWAOptimizer::ParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, const RuntimeConfigurator* configurator)
    : lowered::pass::RuntimeOptimizer(configurator) {
    if (linear_ir->get_config().m_enable_domain_optimization || !linear_ir->is_dynamic())
        return;

    const auto& results = linear_ir->get_results();
    // Note: the feature supports only Subgraphs with single Result,
    // since it is not obvous how to get master shape for multiple Results from this entry point
    if (results.size() != 1)
        return;
    const auto& result = results.back();

    const auto& params = linear_ir->get_parameters();
    // param idx, dim_idx
    std::map<size_t, size_t> params_to_optimize;

    auto check_input_ports = [&params, &params_to_optimize](const std::vector<LoopPort>& ports, bool print = false) {
        for (const auto& port : ports) {
            if (!port.is_processed())
                continue;

            auto expr_to_check = port.get_expr_port()->get_port_connector_ptr()->get_source().get_expr();
            const auto& shape_infer_parents = utils::get_first_parent_shape_infer_expr_seq(expr_to_check);
            if (!shape_infer_parents.empty())
                expr_to_check = shape_infer_parents.back()->get_input_expr_ptr(0);
            if (!ov::is_type<ov::op::v0::Parameter>(expr_to_check->get_node()))
                return false;

            auto found_param = std::find(params.begin(), params.end(), expr_to_check);
            OPENVINO_ASSERT(found_param != params.end(), "find_param didn't found parameter for expr");
            auto first = std::distance(params.begin(), found_param);
            auto second = port.get_dim_idx();
            params_to_optimize[first] = second;
            if (print) {
                std::cout << "[ INFO ] Param to optimize: idx = " << first
                        << ", dim_to_optimize = " << second << "\n";
                std::cout << "\t Loop port: expr_to_check = " << expr_to_check->get_node()->get_friendly_name()
                        << ", dim_idx = " << port.get_dim_idx() << "\n";
            }
        }
        return true;
    };

    auto check_output_ports = [&result](const std::vector<LoopPort>& ports) {
        for (const auto& port : ports) {
            if (!port.is_processed())
                continue;
            const auto consumers = port.get_expr_port()->get_port_connector_ptr()->get_consumers();
            if (!std::all_of(consumers.begin(), consumers.end(), [&result](const auto& consumer) {
                    const auto& node = consumer.get_expr()->get_node();
                    return ov::is_type<ov::snippets::op::LoopEnd>(node) ||
                           (ov::is_type<ov::op::v0::Result>(node) && consumer.get_expr().get() == result.get());
                })) {
                return false;
            }
        }
        return true;
    };

    const auto last_expr = result->get_input_expr_ptr(0);
    const auto outer_loop = last_expr->get_loop_ids().front();
    m_concurrency = linear_ir->get_config().m_min_parallel_work_amount;

    const auto& target_loop = linear_ir->get_loop_manager()->get_loop_info<ExpandedLoopInfo>(outer_loop)->get_unified_loop_info();
    const auto& input_ports = target_loop->get_input_ports();
    const auto& output_ports = target_loop->get_output_ports();
    std::cout << "[ INFO ] ParallelWAOptimizer: input ports check " << (check_input_ports(input_ports) ? "passed" : "failed") << "\n";
    std::cout << "[ INFO ] ParallelWAOptimizer: output ports check " << (check_output_ports(output_ports) ? "passed" : "failed") << "\n";
    // Only loop, whose processed ports are connected to Subgraph inputs/outputs,
    // can be used for parallel work amount optimization
    if (!check_input_ports(input_ports, true) || !check_output_ports(output_ports))
        return;

    std::set<size_t> ids_to_split;
    for (const auto& expr : *linear_ir) {
        const auto loops = expr->get_loop_ids();
        if (loops.empty())
            continue;
        const auto mostouter_loop_id = loops.front();
        const auto mostouter_loop = linear_ir->get_loop_manager()->get_loop_info<ExpandedLoopInfo>(mostouter_loop_id);
        if (mostouter_loop->get_unified_loop_info().get() == target_loop.get()) {
            ids_to_split.insert(mostouter_loop_id);
        }
    }

    for (const auto& id : ids_to_split)
        m_loops_to_split.push_back(linear_ir->get_loop_manager()->get_loop_info<ExpandedLoopInfo>(id));

    std::cout << "[ INFO ] NEW m_loops_to_split size = " << m_loops_to_split.size() << std::endl;

    std::cout << "[ INFO ] ParallelWAOptimizer: loop check passed\n";

    // TODO: collect params to optimize, not ignore ones
    for (size_t i = 0; i < params.size(); ++i) {
        if (params_to_optimize.count(i) == 0)
            m_unsqueezed_params.insert(i);
    }
    std::cout << "[ INFO ] Unsqueezed params: ";
    for (const auto& param : m_unsqueezed_params)
        std::cout << param << " ";
    std::cout << "\n";

    m_dim_M_idces.resize(configurator->get_io_num());
    m_optimized_layouts.resize(configurator->get_io_num());
    for (size_t i = 0; i < configurator->get_io_num(); ++i) {
        const auto& layout = configurator->get_io_descs()[i]->get_layout();
        const size_t fallback_idx = configurator->get_config()->tile_rank - 1;
        OPENVINO_ASSERT(fallback_idx == 1);

        const auto original_dim_idx = params_to_optimize.count(i) ? params_to_optimize.at(i) : fallback_idx;
        const auto dim_idx = i < configurator->get_in_num() ? utils::get_input_dim_idx(layout, original_dim_idx)
                                                            : utils::get_output_dim_idx(layout, original_dim_idx);
        m_dim_M_idces[i] = dim_idx;
        const auto m_idx = i < configurator->get_in_num() ? dim_idx : layout.size() - 1 - fallback_idx;
        m_optimized_layouts[i] = get_updated_order(layout, m_idx);
    }
    std::cout << "\n\n";
}

bool ParallelWAOptimizer::run(const lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ParallelWAOptimizer")
    const auto& config = m_configurator->get_config();
    size_t new_batch_dim, new_kernel_dim;
    bool split_res = MHAParallelWAOptimizer::split(config->master_shape, m_concurrency, new_batch_dim, new_kernel_dim);
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
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov