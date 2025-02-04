// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/parallel_wa_optimizer.hpp"

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

ParallelWAOptimizer::ParallelWAOptimizer(const lowered::LinearIRCPtr& linear_ir, const RuntimeConfigurator* configurator)
    : lowered::pass::RuntimeOptimizer(configurator) {
    if (linear_ir->get_config().m_enable_domain_optimization || !linear_ir->is_dynamic())
        return;

    // const auto brgemms = find_applicable_brgemms(linear_ir);
    // if (brgemms.empty())
    //     return;

    // m_concurrency = linear_ir->get_config().m_min_parallel_work_amount;
    // m_unsqueezed_params = find_unsqueezed_params(linear_ir, brgemms);
    // OPENVINO_ASSERT(!m_unsqueezed_params.empty(), "unsqueezed_params mustn't be empty after initialization");
    // m_loops_to_split = find_loops_to_split(linear_ir, m_unsqueezed_params);

    // m_dim_M_idces.resize(configurator->get_io_num());
    // m_optimized_layouts.resize(configurator->get_io_num());
    // for (size_t i = 0; i < configurator->get_io_num(); ++i) {
    //     const auto& layout = configurator->get_io_descs()[i]->get_layout();
    //     const auto dim_idx = i < configurator->get_in_num() ? utils::get_input_dim_idx(layout, m_dim_M_idx)
    //                                                         : utils::get_output_dim_idx(layout, m_dim_M_idx);
    //     m_dim_M_idces[i] = dim_idx;
    //     const auto m_idx = i < configurator->get_in_num() ? dim_idx : layout.size() - 2;
    //     m_optimized_layouts[i] = get_updated_order(layout, m_idx);
    // }
}

bool ParallelWAOptimizer::run(const lowered::LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ParallelWAOptimizer")
    // const auto& config = m_configurator->get_config();
    // size_t new_batch_dim, new_kernel_dim;
    // bool split_res = split(config->master_shape, m_concurrency, new_batch_dim, new_kernel_dim);
    // std::cout << "Split result: " << split_res << ", Master shape: " << ov::PartialShape(config->master_shape)
    //           << ", Concurrency: " << m_concurrency << ", New batch dimension: " << new_batch_dim
    //           << ", New kernel dimension: " << new_kernel_dim << std::endl;

    // if (!split_res)
    //     return false;
    // auto& master_shape = config->master_shape;
    // *++master_shape.rbegin() = new_kernel_dim;
    // master_shape.insert(master_shape.cbegin() + master_shape.size() - 2, new_batch_dim);
    // m_configurator->update_tensor_rank(master_shape);

    // RuntimeConfigurator::LoopInfoRuntimeParamsMap initialized_info;
    // auto updater = [&](const lowered::LoopInfoPtr& loop_info) {
    //     if (const auto unified_loop_info = ov::as_type_ptr<lowered::UnifiedLoopInfo>(loop_info)) {
    //         if (initialized_info.count(unified_loop_info) == 0) {
    //             if (!ov::is_type<lowered::InnerSplittedUnifiedLoopInfo>(unified_loop_info))
    //                 unified_loop_info->set_work_amount(new_kernel_dim);
    //             snippets::utils::update_data_pointer_shifts(unified_loop_info);
    //             initialized_info[unified_loop_info] = RuntimeConfigurator::get_loop_runtime_params(unified_loop_info);
    //         }
    //     } else if (const auto expanded_loop_info = ov::as_type_ptr<lowered::ExpandedLoopInfo>(loop_info)) {
    //         m_configurator->update_expanded_loop_info(expanded_loop_info, initialized_info);
    //     } else {
    //         OPENVINO_THROW("Failed to update loop info: unknown type!");
    //     }
    // };
    // lowered::LoopInfoSet updated_loops;
    // for (const auto& loop : m_loops_to_split) {
    //     loop->apply(updater, updated_loops);
    // }

    // for (size_t i = 0; i < m_configurator->get_io_num(); ++i) {
    //     config->io_shapes[i] = m_unsqueezed_params.count(i)
    //                     ? unsqueeze_m_dim(config->io_shapes[i], m_dim_M_idces[i])
    //                     : reshape_m_dim(config->io_shapes[i], m_dim_M_idces[i], new_batch_dim, new_kernel_dim);
    // }
    // config->io_layouts = m_optimized_layouts;
    return true;
}
} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov