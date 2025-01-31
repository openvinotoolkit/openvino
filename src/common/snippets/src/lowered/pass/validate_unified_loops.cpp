// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_unified_loops.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils/loop_utils.hpp"
#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void ValidateUnifiedLoops::validate_loop_infos(const LoopManagerPtr& loop_manager) {
    // Already validated vectors of Loop IDs
    std::set<std::vector<size_t>> validated_nested_loops;
    auto is_already_verified = [&validated_nested_loops](const std::vector<size_t>& ids) {
        // If `ids` is subsequence of one of loop_ids in `validated_nested_loops` set,
        // it means that `ids` is already validated too
        for (const auto& loop : validated_nested_loops) {
            if (std::search(ids.cbegin(), ids.cend(), loop.cbegin(), loop.cend()) != ids.cend()) {
                return true;
            }
        }
        return false;
    };

    std::vector<size_t> dim_indexes;

    auto validate_loop_port = [&loop_manager, &dim_indexes, &validated_nested_loops, &is_already_verified](const LoopPort& loop_port) {
        const auto expr = loop_port.get_expr_port()->get_expr();
        const auto& loop_ids = expr->get_loop_ids();
        // If loop_ids of the current port is subsequence of already validated IDs, skip
        if (is_already_verified(loop_ids))
            return;

        dim_indexes.clear();
        dim_indexes.reserve(loop_ids.size());
        // Outer Loop -> Inner Loop
        for (size_t i = 0; i < loop_ids.size(); ++i) {
            const auto id = loop_ids[i];
            const auto dim_idx = loop_manager->get_loop_info(id)->get_dim_idx();
            // if the loop has different dimension indexes, it don't have to meet the split loop related requirements
            if (dim_idx == LoopPort::UNDEFINED_DIM_IDX)
                continue;
            if (i > 0) {
                if (std::find(dim_indexes.cbegin(), dim_indexes.cend(), dim_idx) != dim_indexes.cend()) {
                        OPENVINO_ASSERT(*dim_indexes.rbegin() == dim_idx,
                                        "Incorrect Loop ID configuration: the Loops with splitted dimension should be successively nested");
                        OPENVINO_ASSERT(loop_manager->get_loop_info(loop_ids[i - 1])->get_increment() == loop_manager->get_loop_info(id)->get_work_amount(),
                                        "Incorrect Loop ID configuration: the Loops with splitted dimension should be successively nested");
                }
            }
            dim_indexes.push_back(dim_idx);
        }
        validated_nested_loops.insert(loop_ids);
    };

    for (const auto& pair : loop_manager->get_map()) {
        const auto& loop_info = ov::as_type_ptr<UnifiedLoopInfo>(pair.second);
        OPENVINO_ASSERT(loop_info, "ValidateUnifiedLoops expects only UnifiedLoopInfo in LoopManager");
        loop_info->iterate_through_ports(validate_loop_port);

        // Validate that iteration dimension is broadcastable
        std::set<size_t> unique_dimensions;
        loop_info->iterate_through_ports([&unique_dimensions](const LoopPort& loop_port) {
            if (loop_port.is_processed()) {
                const auto is_input = loop_port.get_expr_port()->get_type() == ExpressionPort::Input;
                const auto planar_shape = is_input ? ov::snippets::utils::get_planar_vdims(*loop_port.get_expr_port())
                                                   : ov::snippets::utils::get_preordered_vdims(*loop_port.get_expr_port());
                const auto& dim = *(planar_shape.rbegin() + loop_port.get_dim_idx());
                // Since dim == 1 can be broadcasted to any value, it's not necessary to add it to unique dims
                if (!utils::is_dynamic_value(dim) && dim != 1)
                    unique_dimensions.insert(dim);
            }
        });
        OPENVINO_ASSERT(unique_dimensions.size() <= 1,
                        "Loop ports have incompatible dimensions, by which the loop iterates");
    }
}

void ValidateUnifiedLoops::validate_loop_port_presence(const LinearIR& linear_ir) {
    auto validate_loop_port = [](const ExpressionPort& expr_port, const LoopInfoPtr& loop_info, size_t loop_id) {
        if (utils::should_be_loop_port(expr_port, loop_id)) {
            OPENVINO_ASSERT(loop_info->is_loop_port(expr_port),
                            "Expression port with idx ", expr_port.get_index(), " with node ",
                            expr_port.get_expr()->get_node()->get_friendly_name(), " is not Loop port but should be!");
        } else {
            OPENVINO_ASSERT(!loop_info->is_loop_port(expr_port),
                            "Expression port with idx ", expr_port.get_index(), " with node ",
                            expr_port.get_expr()->get_node()->get_friendly_name(), " is Loop port but should not be!");
        }
    };

    const auto& loop_manager = linear_ir.get_loop_manager();
    for (const auto& expr : linear_ir) {
        const auto& op = expr->get_node();
        if (ov::is_type<ov::snippets::op::LoopBase>(op))
            continue;

        for (const auto& loop_id : expr->get_loop_ids()) {
            const auto& loop_info = loop_manager->get_loop_info(loop_id);

            for (size_t i = 0; i < expr->get_input_count(); ++i)
                validate_loop_port(expr->get_input_port(i), loop_info, loop_id);

            for (size_t i = 0; i < expr->get_output_count(); ++i)
                validate_loop_port(expr->get_output_port(i), loop_info, loop_id);
        }
    }
}

bool ValidateUnifiedLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateUnifiedLoops")
    if (linear_ir.empty())
        return false;

    validate_loop_infos(linear_ir.get_loop_manager());
    validate_loop_port_presence(linear_ir);

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
