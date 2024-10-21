// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/split_loops.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/pass/fuse_loops.hpp"
#include "snippets/lowered/pass/init_loops.hpp"
#include "snippets/lowered/pass/iter_handler.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/loop_utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool SplitLoops::can_be_split(const UnifiedLoopInfoPtr& loop_to_split, const UnifiedLoopInfoPtr& loop_to_fuse) {
    OPENVINO_ASSERT(loop_to_split != nullptr && loop_to_fuse != nullptr, "LoopInfo is nullptr!");
    const auto current_dim_idx = loop_to_split->get_dim_idx();
    const auto parent_dim_idx = loop_to_fuse->get_dim_idx();
    const auto& handlers = loop_to_split->get_handlers();
    const bool equal_dim_idxes = current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX && current_dim_idx == parent_dim_idx;
    const bool only_main_body = handlers.get_passes<SpecificLoopIterType::FIRST_ITER>().empty() &&
                                handlers.get_passes<SpecificLoopIterType::LAST_ITER>().empty();
    std::cout << "loop_to_split->get_work_amount():" << loop_to_split->get_work_amount() << std::endl;
    std::cout << "loop_to_fuse->get_work_amount():" << loop_to_fuse->get_work_amount() << std::endl;
    std::cout << "loop_to_split->get_increment():" << loop_to_split->get_increment() << std::endl;
    std::cout << "loop_to_fuse->get_increment():" << loop_to_fuse->get_increment() << std::endl;
    std::cout << "current_dim_idx:" << current_dim_idx << std::endl;
    std::cout << "parent_dim_idx:" << parent_dim_idx << std::endl;
    std::cout << "equal_dim_idxes:" << equal_dim_idxes << std::endl;
    std::cout << "only_main_body:" << only_main_body << std::endl;
    return loop_to_split->get_work_amount() == loop_to_fuse->get_work_amount() &&
           loop_to_split->get_increment() != loop_to_fuse->get_increment() && equal_dim_idxes && only_main_body;
}

bool SplitLoops::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::SplitLoops")
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool loop_was_split = false;
    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        bool loop_was_split_expr = false;
        const auto& expr = *expr_it;
        const auto loop_ids = expr->get_loop_ids();
        if (loop_ids.empty())
            continue;
        std::cout << "---------------------" << std::endl;
        std::cout << "expr:" << expr->get_node()->get_friendly_name() << std::endl;
        for (size_t i = 0; i < loop_ids.size(); i++) {
            std::cout << "loop_id:" << loop_ids[i] << std::endl;
        }

        // Ticket: 113755
        // Note: we currently consider only the outermost loops or inner loop if outers are splited and fused.
        // Splitting could also be done in a more general case, but the splitted loop and its parent must always
        // be in the same set of outer loops. Otherwise they won't be fused.
        const auto& loop_depth = loop_ids.size();
        size_t block_loop_axis = SIZE_MAX;
        std::cout << "loop_depth:" << loop_depth << std::endl;
        for (size_t d = 0; d < loop_depth; d++) {
            // if outter loop is not split and fused, inner loop should stop split
            std::cout << "d:" << d << std::endl;
            // if (d > 0 && !loop_was_split_expr) {
            //     std::cout << "d > 0 && !loop_was_split_expr" << std::endl;
            //     break;
            // }
            const auto& loop_id = loop_ids[d]; // loop_ids[loop_was_split ? d+1 : d];
            const auto loop = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
            size_t exp_loop_dim_idx = loop->get_dim_idx();
            std::cout << "loop->get_input_ports():" << loop->get_input_ports().size() << std::endl;
            for (const auto& input_port : loop->get_input_ports()) {  // fused outter in one port, inner in another port?
                const auto& parent_port = input_port.expr_port->get_port_connector_ptr()->get_source();
                const auto& parent_expr = parent_port.get_expr();
                const auto& parent_loop_ids = parent_expr->get_loop_ids();
                std::cout << "parent_expr:" << parent_expr->get_node()->get_friendly_name() << std::endl;
                for (size_t i = 0; i < parent_loop_ids.size(); i++) {
                    std::cout << "parent_loop_ids:" << parent_loop_ids[i] << std::endl;
                }
                if (parent_loop_ids.empty())
                    continue;
                // if (parent_loop_ids.size() != loop_depth) {
                //     continue;
                // }

                // find parent loop based on dim index
                std::cout << "exp_loop_dim_idx:" << exp_loop_dim_idx << std::endl;
                const auto parent_loop_idx = std::find_if(parent_loop_ids.begin(), parent_loop_ids.end(), [&](size_t loop_id){
                    const auto parent_loop = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
                    std::cout << "parent_loop->get_dim_idx():" << parent_loop->get_dim_idx() << std::endl;
                    return parent_loop->get_dim_idx() == exp_loop_dim_idx;
                });
                if (parent_loop_idx == parent_loop_ids.end()) {
                    std::cout << "parent_loop_idx == parent_loop_ids.end()" << std::endl;
                    continue;
                } else {
                    std::cout << "parent_loop_idx != parent_loop_ids.end()" << std::endl;
                }
                const auto& parent_loop_id = *parent_loop_idx;
                std::cout << "find parent_loop_id:" << parent_loop_id << std::endl;
                // const auto& parent_loop_id = parent_loop_ids[d];
                const auto parent_loop = loop_manager->get_loop_info<UnifiedLoopInfo>(parent_loop_id);

                const bool split_parent = parent_loop->get_increment() < loop->get_increment();
                const auto upper_loop = std::make_shared<UnifiedLoopInfo>(*parent_loop);
                const auto lower_loop = std::make_shared<UnifiedLoopInfo>(*loop);
                if (split_parent)
                    upper_loop->set_increment(loop->get_increment());
                else
                    lower_loop->set_increment(parent_loop->get_increment());

                const auto& loop_to_split = split_parent ? parent_loop : loop;
                const auto& loop_to_fuse = !split_parent ? parent_loop : loop;
                // We don't split loop which are not compatible with parent loop because such loops will not be fused
                std::cout << "can_be_fused(upper_loop, lower_loop):" << FuseLoops::can_be_fused(upper_loop, lower_loop) << std::endl;
                std::cout << "can_be_split(loop_to_split, loop_to_fuse):" << can_be_split(loop_to_split, loop_to_fuse) << std::endl;
                if (FuseLoops::can_be_fused(upper_loop, lower_loop) && can_be_split(loop_to_split, loop_to_fuse)) {
                    const auto& loop_to_split_id = split_parent ? parent_loop_id : loop_id;
                    // find first loop with outer loop_dim(x+1), insert after iter(before next of iter)
                    const auto& outer_loop_idx = exp_loop_dim_idx + 1;  // dim 0 means innermost
                    std::cout << "exp_loop_dim_idx:" << exp_loop_dim_idx << std::endl;
                    std::cout << "outer_loop_idx:" << outer_loop_idx << std::endl;
                    auto outer_loop_idx_iter = std::find_if(loop_ids.cbegin(), loop_ids.cend(), [&](size_t loop_id) {
                        const auto loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_id);
                        return loop_info->get_dim_idx() == outer_loop_idx;
                    });
                    if (outer_loop_idx_iter == loop_ids.end()) {
                        block_loop_axis = SIZE_MAX;   // for exp_loop_dim_idx is already outermost
                    } else {
                        std::cout << "outer_loop_idx_iter:" << *outer_loop_idx_iter << std::endl;
                        outer_loop_idx_iter++;
                        block_loop_axis = *outer_loop_idx_iter;
                    }
                    std::cout << "block_loop_axis:" << block_loop_axis << std::endl;

                    size_t id = split(linear_ir, loop_to_split_id, loop_to_fuse->get_increment(), block_loop_axis);
                    // if (block_loop_axis == SIZE_MAX) {
                    //     block_loop_axis = id;  // always insert before first block inner loop, after first mark loop, id changed
                    // }
                    loop_was_split = true;
                    loop_was_split_expr = true;
                    std::cout << "splited loop id:" << loop_to_split_id << std::endl;
                    break;
                }
            }
        }
    }
    // Ticket: 113666
    // FuseLoops pass is explicitly run here in order to avoid unnecessary computations
    // in case if loops are not split but FuseLoops is registered in pass manager after SplitLoops
    if (loop_was_split)
        FuseLoops().run(linear_ir, begin, end);
    return loop_was_split;
}

size_t SplitLoops::split(LinearIR& linear_ir, size_t loop_to_split_id, size_t outer_increment, size_t loop_position) {
    const auto& loop_manager = linear_ir.get_loop_manager();

    const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(loop_to_split_id);
    const auto loop_bounds = LoopManager::get_loop_bounds(linear_ir, loop_to_split_id,
                                                          inner_loop_info->get_input_ports(),
                                                          inner_loop_info->get_output_ports());
    std::cout << "loop_bounds.first:" << (*loop_bounds.first)->get_node()->get_friendly_name() << std::endl;
    std::cout << "loop_bounds.second:" << (*loop_bounds.second)->get_node()->get_friendly_name() << std::endl;
    auto outer_loop_bounds_first = loop_bounds.first;
    auto outer_loop_bounds_second = loop_bounds.second;
    auto in_ports = inner_loop_info->get_input_ports();
    auto& out_ports = inner_loop_info->get_output_ports();
    if (inner_loop_info->get_dim_idx() == 0) {
        for (auto expr_it = outer_loop_bounds_first; expr_it != outer_loop_bounds_second; expr_it++) {
            auto expr = *expr_it;
            std::cout << "expr:" << expr->get_node()->get_friendly_name() << std::endl;
            if (expr->get_output_count() != 1)
                continue;
            auto expr_out_port = expr->get_output_port(0);
            bool check_next = true;
            const auto& expr_loop = expr->get_loop_ids();
            bool extend = false;
            while (check_next) {
                if (expr->get_output_count() != 1)
                    break;
                const auto& child_expr = expr->get_output_port_connector(0)->get_consumers().begin()->get_expr();
                if (child_expr->get_input_count() != 1)
                    break;
                bool is_inside = false;
                for (auto expr_check = outer_loop_bounds_first; expr_check != outer_loop_bounds_second; expr_check++) {
                    if (*expr_check == child_expr) {
                        is_inside = true;
                        break;
                    }
                }
                std::cout << "is_inside:" << is_inside << std::endl;
                if (is_inside)
                    break;
                const auto& child_expr_loop = child_expr->get_loop_ids();
                // have common outer loop id, child w/o inner most loop
                for (auto i = 0; i < std::min(expr_loop.size(), child_expr_loop.size()); i++) {
                    if (expr_loop[i] != child_expr_loop[i]) {
                        break;
                    }
                }
                const auto& last_loop_dim = loop_manager->get_loop_info<UnifiedLoopInfo>(child_expr_loop.back())->get_dim_idx();
                if ((expr_loop.size() - child_expr_loop.size() == 1) && last_loop_dim != 0) {
                    outer_loop_bounds_second = linear_ir.find(child_expr);
                    extend = true;
                    std::cout << "extend........" << std::endl;
                } else {
                    break;
                }
                expr = child_expr;
            }
            if (extend) {
                // if expr port is loop port, replace to new child output port.
                auto loop_port = std::find_if(out_ports.begin(), out_ports.end(), [&](LoopPort loop_ports) {
                    return *loop_ports.expr_port.get() == expr_out_port;
                });
                if (loop_port != out_ports.end()) {
                    inner_loop_info->replace_with_new_ports(expr_out_port, {(*outer_loop_bounds_second)->get_output_port(0)});
                }
                // outer_loop_bounds_second is point to last expr, move to next of last
                outer_loop_bounds_second++;
            }
            auto expr_for_parent = *expr_it;
        }
        // extend parent
        for (auto expr_it = outer_loop_bounds_first; expr_it != outer_loop_bounds_second; expr_it++) {
            auto expr = *expr_it;
            std::cout << "expr_for_parent:" << expr->get_node()->get_friendly_name() << std::endl;
            // if (expr->get_input_count() != 1)
            //     continue;
            auto in_num = expr->get_input_count();
            if (in_num > 2 || in_num < 1)
               continue;
            auto expr_in_port = expr->get_input_port(in_num-1);
            bool check_next = true;
            const auto& expr_loop = expr->get_loop_ids();
            bool extend = false;
            while (check_next) {
                // if (expr->get_input_count() != 1)
                //     break;
                in_num = expr->get_input_count();
                if (in_num > 2 || in_num < 1)
                    break;
                auto parent_expr = expr->get_input_port_connector(in_num-1)->get_source().get_expr();
                std::cout << "parent_expr:" << parent_expr->get_node()->get_friendly_name() << std::endl;
                if (parent_expr->get_output_count() != 1)
                    break;
                bool is_inside = false;
                for (auto expr_check = outer_loop_bounds_first; expr_check != outer_loop_bounds_second; expr_check++) {
                    if (*expr_check == parent_expr) {
                        is_inside = true;
                        break;
                    }
                }
                std::cout << "is_inside:" << is_inside << std::endl;
                if (is_inside)
                    break;
                const auto& parent_expr_loop = parent_expr->get_loop_ids();
                // have common outer loop id, child w/o inner most loop
                for (auto i = 0; i < std::min(expr_loop.size(), parent_expr_loop.size()); i++) {
                    if (expr_loop[i] != parent_expr_loop[i]) {
                        break;
                    }
                }
                const auto& last_loop_dim = loop_manager->get_loop_info<UnifiedLoopInfo>(parent_expr_loop.back())->get_dim_idx();
                if ((expr_loop.size() - parent_expr_loop.size() == 1) && last_loop_dim != 0) {
                    outer_loop_bounds_first = linear_ir.find(parent_expr);
                    extend = true;
                    std::cout << "extend........" << std::endl;
                } else {
                    break;
                }
                expr = parent_expr;
            }
            if (extend) {
                // if expr port is loop port, replace to new child output port.
                auto loop_port = std::find_if(in_ports.begin(), in_ports.end(), [&](LoopPort loop_ports) {
                    return *loop_ports.expr_port.get() == expr_in_port;
                });
                if (loop_port != out_ports.end()) {
                    // inner_loop_info->replace_with_new_ports(expr_in_port, {(*outer_loop_bounds_first)->get_output_port(0)});
                    inner_loop_info->replace_with_new_ports(expr_in_port, {});
                }
            }
        }
    }

    // if ((*loop_bounds.second)->get_node()->get_friendly_name() == "HorizonMax_4330" ||
    //     (*loop_bounds.second)->get_node()->get_friendly_name() == "HorizonSum_4335") {
    //     auto expr_port = (*outer_loop_bounds_second)->get_input_port_connector(0)->get_source();
    //     auto loop_port = std::find_if(out_ports.begin(), out_ports.end(), [&](LoopPort loop_ports) {
    //         return *loop_ports.expr_port.get() == expr_port;
    //     });
    //     if (loop_port != out_ports.end()) {
    //         std::cout << "out_ports.erase(loop_port) HorizonMax_4330;" << std::endl;
    //         inner_loop_info->replace_with_new_ports(expr_port, {(*loop_bounds.second)->get_output_port(0)});
    //         // out_ports.erase(loop_port);
    //         // LoopPort new = std::make_shared<LoopPort>((*loop_bounds.second)->get_output_port(0));
    //         // out_ports.insert(LoopPort{(*loop_bounds.second)->get_output_port(0)});
    //     }
    //     outer_loop_bounds_second++;
    // }
    // if ((*loop_bounds.first)->get_node()->get_friendly_name() == "Fill_4328") {
    //     outer_loop_bounds_first--;
    //     // outer_loop_bounds_first is Fill_4327 now
    //     auto expr_port = (*outer_loop_bounds_first)->get_output_port_connector(0)->get_consumers().begin();
    //     auto loop_port = std::find_if(in_ports.begin(), in_ports.end(), [&](LoopPort loop_ports) {
    //         return *loop_ports.expr_port.get() == *expr_port;
    //     });
    //     if (loop_port != in_ports.end()) {
    //         std::cout << "in_ports.erase(loop_port) Fill_4327;" << std::endl;
    //         in_ports.erase(loop_port);
    //     }
    //     outer_loop_bounds_first--;
    // }
    const auto outer_loop_id = loop_manager->mark_loop(outer_loop_bounds_first, outer_loop_bounds_second, inner_loop_info->get_work_amount(),
                                                       outer_increment, inner_loop_info->get_dim_idx(),
                                                       in_ports, out_ports,
                                                       false, true, loop_position);
    const auto& outer_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(outer_loop_id);

    const auto& inner_splitted_loop_info =
        std::make_shared<InnerSplittedUnifiedLoopInfo>(inner_loop_info->get_increment(), inner_loop_info->get_input_ports(),
                                                       inner_loop_info->get_output_ports(), inner_loop_info->get_input_port_descs(),
                                                       inner_loop_info->get_output_port_descs(), inner_loop_info->get_handlers(),
                                                       outer_loop_info);
    size_t new_inner = loop_manager->replace_with_new_loop(linear_ir, loop_bounds.first, loop_bounds.second, inner_splitted_loop_info, loop_to_split_id);
    // std::cout << "new outer_loop_id" << outer_loop_id  << " new_inner:" << new_inner << std::endl;
    // if (!outer_loop_info->get_handlers().get_passes<SpecificLoopIterType::FIRST_ITER>().empty()) {
    //     outer_loop_info->register_pass_to_handler<SpecificLoopIterType::FIRST_ITER, TransformInnerSplitLoop>();
    // }
    // outer_loop_info->register_pass_to_handler<SpecificLoopIterType::MAIN_BODY, TransformInnerSplitLoop>();
    // outer_loop_info->register_pass_to_handler<SpecificLoopIterType::LAST_ITER, TransformInnerSplitLoop>();

    std::cout << "new outer:" << outer_loop_id << std::endl;
    std::cout << "new_inner:" << new_inner << std::endl;

    return new_inner;
}

namespace {
InnerSplittedUnifiedLoopInfoPtr make_own_inner_splitted_unified_loop_info(const ExpandedLoopInfoPtr& inner_expanded,
                                                                          const ExpandedLoopInfoPtr& outer_expanded,
                                                                          const InnerSplittedUnifiedLoopInfoPtr& existing_inner_unified) {
    const auto loop_info =
        std::make_shared<InnerSplittedUnifiedLoopInfo>(inner_expanded->get_increment(), inner_expanded->get_input_ports(),
                                                       inner_expanded->get_output_ports(), existing_inner_unified->get_input_port_descs(),
                                                       existing_inner_unified->get_output_port_descs(), existing_inner_unified->get_handlers(),
                                                       outer_expanded);
    ov::snippets::utils::update_runtime_parameters(loop_info);
    return loop_info;
}
ExpandedLoopInfoPtr make_own_inner_splitted_expanded_loop_info(const ExpandedLoopInfoPtr& inner_expanded,
                                                               const InnerSplittedUnifiedLoopInfoPtr& inner_unified) {
    return std::make_shared<ExpandedLoopInfo>(inner_unified->get_work_amount(), inner_unified->get_increment(),
                                              inner_unified->get_input_ports(), inner_unified->get_output_ports(),
                                              inner_unified->get_ptr_increments(),
                                              inner_unified->get_finalization_offsets(),
                                              inner_unified->get_data_sizes(), inner_expanded->get_type(),
                                              inner_unified, inner_expanded->is_evaluate_once());
}
}  // namespace

bool SplitLoops::TransformInnerSplitLoop::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    OPENVINO_ASSERT(end != linear_ir.cend(), "Incorrect LinearIR range for processing");
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    OPENVINO_ASSERT(loop_end, "the last operation in range must be LoopEnd");

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& outer_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(loop_end->get_id());
    const auto current_dim_idx = outer_loop_info->get_dim_idx();
    OPENVINO_ASSERT(current_dim_idx != LoopInfo::UNDEFINED_DIM_IDX,
                    "Outer splitted loop unexpectedly iterates by several dimension indices");

    bool modified = false;
    for (auto it = begin; it != end; ++it) {
        const auto& expr = *it;
        const auto inner_loop_end = ov::as_type_ptr<op::LoopEnd>(expr->get_node());
        if (!inner_loop_end)
            continue;

        // There is already ExpandedLoopInfo
        const auto& inner_expanded_loop_info = loop_manager->get_loop_info<ExpandedLoopInfo>(inner_loop_end->get_id());
        const auto inner_unified_loop_info = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(inner_expanded_loop_info->get_unified_loop_info());
        if (!inner_unified_loop_info || inner_unified_loop_info->get_outer_splitted_loop_info() != outer_loop_info->get_unified_loop_info())
            continue;

        OPENVINO_ASSERT(current_dim_idx == inner_unified_loop_info->get_dim_idx(), "Incorrect processing dim index of splitted loops");
        OPENVINO_ASSERT(inner_expanded_loop_info->get_type() == SpecificLoopIterType::MAIN_BODY, "InnerSplittedLoop must be Main Body of loop");

        // We have to make a new UnifiedLoopInfo to distinguish it from other unified loops in other specific iterations of outer loop.
        const auto inner_splitted_unified_loop_info = make_own_inner_splitted_unified_loop_info(inner_expanded_loop_info, outer_loop_info,
                                                                                                inner_unified_loop_info);

        // We have to replace existing ExpandedLoopInfo with new one to have the own InnerSplittedUnifiedLoopInfo and
        // distinguish it from other expanded loops in other specific iterations of outer loop.
        const auto new_expanded_inner_loop_info = make_own_inner_splitted_expanded_loop_info(inner_expanded_loop_info, inner_splitted_unified_loop_info);
        const auto inner_begin = linear_ir.find_before(it, linear_ir.get_expr_by_node(inner_loop_end->get_loop_begin()));
        const auto new_id = loop_manager->replace_with_new_loop(linear_ir, inner_begin, std::next(it), new_expanded_inner_loop_info, inner_loop_end->get_id());

        // [147894] : Update inner LoopEnd expression
        inner_loop_end->set_id(new_id);
        inner_loop_end->set_work_amount(new_expanded_inner_loop_info->get_work_amount());
        inner_loop_end->set_increment(new_expanded_inner_loop_info->get_increment());
        inner_loop_end->set_finalization_offsets(new_expanded_inner_loop_info->get_finalization_offsets());
    }
    return modified;
}

std::shared_ptr<pass::PassBase> SplitLoops::TransformInnerSplitLoop::merge(const std::shared_ptr<pass::PassBase>& other) {
    return !other || ov::is_type<TransformInnerSplitLoop>(other) ? std::make_shared<TransformInnerSplitLoop>() : nullptr;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov