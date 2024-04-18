// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_unified_loops.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool ValidateUnifiedLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateUnifiedLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();

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

    auto validate_loop_ports = [&loop_manager, &dim_indexes, &validated_nested_loops, &is_already_verified](const std::vector<LoopPort>& loop_ports) {
        for (const auto& loop_port : loop_ports) {
            const auto expr = loop_port.expr_port->get_expr();
            const auto& loop_ids = expr->get_loop_ids();
            // If loop_ids of the current port is subsequence of already validated IDs, skip
            if (is_already_verified(loop_ids))
                continue;

            dim_indexes.clear();
            dim_indexes.reserve(loop_ids.size());
            // Outer Loop -> Inner Loop
            for (size_t i = 0; i < loop_ids.size(); ++i) {
                const auto id = loop_ids[i];
                const auto dim_idx = loop_manager->get_loop_info(id)->get_dim_idx();
                // if the loop has different dimension indexes, it don't have to meet the split loop related requirements
                if (dim_idx == LoopInfo::UNDEFINED_DIM_IDX)
                    continue;
                if (std::find(dim_indexes.cbegin(), dim_indexes.cend(), dim_idx) != dim_indexes.cend()) {
                    OPENVINO_ASSERT(*dim_indexes.rbegin() == dim_idx,
                                    "Incorrect Loop ID configuration: the Loops with splitted dimension should be successively nested");
                    OPENVINO_ASSERT(loop_manager->get_loop_info(loop_ids[i - 1])->get_increment() == loop_manager->get_loop_info(id)->get_work_amount(),
                                    "Incorrect Loop ID configuration: the Loops with splitted dimension should be successively nested");
                }
                dim_indexes.push_back(dim_idx);
            }
            validated_nested_loops.insert(loop_ids);
        }
    };

    auto add_ports_dims_to_unique_dims = [](const std::vector<LoopPort>& loop_ports, std::set<size_t>& unique_dims, bool is_entry) {
        for (const auto& loop_port : loop_ports) {
            if (!loop_port.is_incremented)
                continue;
            const auto planar_shape = is_entry ? ov::snippets::utils::get_planar_vdims(*loop_port.expr_port)
                                               : ov::snippets::utils::get_preordered_vdims(*loop_port.expr_port);
            const auto& dim = *(planar_shape.rbegin() + loop_port.dim_idx);
            // Since dim == 1 can be broadcasted to any value, it's not necessary to add it to unique dims
            if (!utils::is_dynamic_value(dim) && dim != 1)
                unique_dims.insert(dim);
        }
    };

    for (const auto& pair : loops) {
        const auto& loop_info = std::dynamic_pointer_cast<UnifiedLoopInfo>(pair.second);
        OPENVINO_ASSERT(loop_info,
                        "ValidateUnifiedLoops expects only UnifiedLoopInfo in LoopManager");
        const auto& entry_points = loop_info->get_entry_points();
        const auto& exit_points = loop_info->get_exit_points();
        validate_loop_ports(entry_points);
        validate_loop_ports(exit_points);

        std::set<size_t> unique_dimensions;
        add_ports_dims_to_unique_dims(entry_points, unique_dimensions, true);
        add_ports_dims_to_unique_dims(exit_points, unique_dimensions, false);
        OPENVINO_ASSERT(unique_dimensions.size() <= 1,
                        "Loop ports have incompatible dimensions, by which the loop iterates");
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
