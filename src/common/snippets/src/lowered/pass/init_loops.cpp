// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
void filter_ports(LinearIR& linear_ir,
                  std::vector<LinearIR::LoopManager::LoopPoint>& loop_entries, std::vector<LinearIR::LoopManager::LoopPoint>& loop_exits) {
    std::vector<LinearIR::LoopManager::LoopPoint> new_loop_entries;
    std::vector<LinearIR::LoopManager::LoopPoint> new_loop_exits;
    new_loop_entries.reserve(loop_entries.size());
    new_loop_exits.reserve(loop_exits.size());

    for (const auto& loop_entry_point : loop_entries) {
        const auto& expr = loop_entry_point.port.get_expr();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
        if (ma && ma->is_memory_access_input_port(loop_entry_point.port.get_index())) {
            new_loop_entries.push_back(loop_entry_point);
        }
    }

    for (const auto& loop_exit_point : loop_exits) {
        const auto& expr = loop_exit_point.port.get_expr();
        const auto ma = ov::as_type_ptr<op::MemoryAccess>(expr->get_node());
        if (ma && ma->is_memory_access_output_port(loop_exit_point.port.get_index())) {
            new_loop_exits.push_back(loop_exit_point);
        }
    }

    loop_entries = new_loop_entries;
    loop_exits = new_loop_exits;
}

int64_t get_dim_stride(const size_t dim, const std::vector<size_t>& layout, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim)
            break;
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
void add_stride_from_splitted_loops(int64_t& increment, const LinearIR::LoopManagerPtr& loop_manager,
                                    const std::vector<size_t>& loop_ids, size_t loop_id, size_t dim_idx) {
    // Example, shape = [3, 384, 64], loop_ids = [0, 1, 2]
    // Loop Info:                            | Pointer increments:
    // - 0: work_amount = 12, dim_idx = 1    | 1 x 64 x 32 - 32 is work_amount of inner splitted Loop
    // - 1: work_amount = 32, dim_idx = 1    | 1 x 64
    // - 2: work_amount = 64, dim_idx = 0    | 1

    // Firstly, we find all Loop IDs with the same dimension index.
    // The Loop Info's with the same dimension index mean that these Loops split this dimension together.
    // It's possible in Brgemm Blocking by M, for example
    std::vector<size_t> splitted_loops;
    // Inner -> Outer
    for (auto it = loop_ids.rbegin(); it != loop_ids.rend(); ++it) {
        const auto id = *it;
        if (loop_manager->get_loop_info(id)->dim_idx == dim_idx)
            splitted_loops.push_back(id);
    }
    // Secondly, we added work amount of inner splitted Loops
    for (auto id : splitted_loops) {
        if (id == loop_id)
            break;
        increment *= loop_manager->get_loop_info(id)->work_amount;
    }
}
}  // namespace

InitLoops::InitLoops() : Pass() {}

std::vector<int64_t> InitLoops::init_ptr_increments(std::vector<LinearIR::LoopManager::LoopPoint>& loop_inputs,
                                                    std::vector<LinearIR::LoopManager::LoopPoint>& loop_outputs,
                                                    const LinearIR::LoopManagerPtr& loop_manager,
                                                    size_t loop_id, size_t work_amount, size_t dim_idx) {
    std::vector<int64_t> ptr_increments;
    ptr_increments.reserve(loop_inputs.size() + loop_outputs.size());

    for (auto& loop_input : loop_inputs) {
        const auto& port = loop_input.port;
        // For strides we have to use layout from source since source writes data by special rules
        const auto source = *port.get_connected_ports().begin();
        const auto loop_ids = port.get_expr()->get_loop_ids();
        const auto& layout = port.get_descriptor_ptr()->get_layout();
        const auto& shape = port.get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        // If ptr increment has not been manually set
        if (loop_input.ptr_increment == LinearIR::LoopManager::LoopPoint::UNDEFINED) {
            loop_input.ptr_increment = 0;
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(shape[dim] == 1 && work_amount != 1)) {
                loop_input.ptr_increment = get_dim_stride(dim, source.get_descriptor_ptr()->get_layout(), shape);
                add_stride_from_splitted_loops(loop_input.ptr_increment, loop_manager, loop_ids, loop_id, dim_idx);
            }
        }
        ptr_increments.push_back(loop_input.ptr_increment);
    }

    for (auto& loop_output : loop_outputs) {
        const auto& port = loop_output.port;
        const auto loop_ids = port.get_expr()->get_loop_ids();
        const auto& layout = port.get_descriptor_ptr()->get_layout();
        const auto& shape = port.get_descriptor_ptr()->get_shape();
        const auto& dim = *(layout.rbegin() + dim_idx);
        // If ptr increment has not been manually set
        if (loop_output.ptr_increment == LinearIR::LoopManager::LoopPoint::UNDEFINED) {
            loop_output.ptr_increment = 0;
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(shape[dim] == 1 && work_amount != 1)) {
                loop_output.ptr_increment = get_dim_stride(dim, layout, shape);
                add_stride_from_splitted_loops(loop_output.ptr_increment, loop_manager, loop_ids, loop_id, dim_idx);
            }
        }
        ptr_increments.push_back(loop_output.ptr_increment);
    }
    return ptr_increments;
}

std::vector<int64_t> InitLoops::init_finalization_offsets(std::vector<LinearIR::LoopManager::LoopPoint>& loop_inputs,
                                                          std::vector<LinearIR::LoopManager::LoopPoint>& loop_outputs,
                                                          size_t work_amount) {
    std::vector<int64_t> finalization_offsets;
    finalization_offsets.reserve(loop_inputs.size() + loop_outputs.size());

    for (auto& loop_input : loop_inputs) {
        // If finalization offset has not been manually set
        if (loop_input.finalization_offset == LinearIR::LoopManager::LoopPoint::UNDEFINED) {
            loop_input.finalization_offset = -1 * loop_input.ptr_increment * work_amount;
        }
        finalization_offsets.push_back(loop_input.finalization_offset);
    }
    for (auto& loop_output : loop_outputs) {
        // If finalization offset has not been manually set
        if (loop_output.finalization_offset == LinearIR::LoopManager::LoopPoint::UNDEFINED) {
            loop_output.finalization_offset = -1 * loop_output.ptr_increment * work_amount;
        }
        finalization_offsets.push_back(loop_output.finalization_offset);
    }
    return finalization_offsets;
}

std::vector<int64_t> InitLoops::init_element_type_sizes(const std::vector<LinearIR::LoopManager::LoopPoint>& loop_inputs,
                                                        const std::vector<LinearIR::LoopManager::LoopPoint>& loop_outputs) {
    std::vector<int64_t> element_types;
    element_types.reserve(loop_inputs.size() + loop_outputs.size());
    for (const auto& in : loop_inputs) {
        const auto& port = in.port;
        element_types.push_back(port.get_expr()->get_node()->get_input_element_type(port.get_index()).size());
    }
    for (const auto& out : loop_outputs) {
        const auto& port = out.port;
        element_types.push_back(port.get_expr()->get_node()->get_output_element_type(port.get_index()).size());
    }
    return element_types;
}

void InitLoops::insertion(LinearIR& linear_ir, const LinearIR::LoopManagerPtr& loop_manager, size_t loop_id, bool has_outer_loop) {
    const auto loop_info = loop_manager->get_loop_info(loop_id);
    auto loop_entries = loop_info->entry_points;
    auto loop_exits = loop_info->exit_points;
    const auto work_amount = loop_info->work_amount;
    const auto work_amount_increment = loop_info->increment;
    const auto dim_idx = loop_info->dim_idx;

    LinearIR::constExprIt loop_begin_pos, loop_end_pos;
    loop_manager->get_loop_bounds(linear_ir, loop_id, loop_begin_pos, loop_end_pos);

    filter_ports(linear_ir, loop_entries, loop_exits);

    const auto ptr_increments = init_ptr_increments(loop_entries, loop_exits, loop_manager, loop_id, work_amount, dim_idx);
    const auto finalization_offsets = init_finalization_offsets(loop_entries, loop_exits, work_amount);
    const auto io_data_sizes = init_element_type_sizes(loop_entries, loop_exits);

    const auto& loop_begin = std::make_shared<op::LoopBegin>();
    const auto& loop_begin_expr = linear_ir.create_expression(loop_begin, std::vector<PortConnectorPtr>{});
    linear_ir.insert(loop_begin_pos, loop_begin_expr);

    const auto& loop_end = std::make_shared<op::LoopEnd>(
            loop_begin->output(0), work_amount, work_amount_increment, ptr_increments, finalization_offsets,
            io_data_sizes, loop_entries.size(), loop_exits.size());
    loop_end->has_outer_loop = has_outer_loop;

    std::vector<PortConnectorPtr> loop_end_inputs;
    for (const auto& expr_point : loop_entries)
        loop_end_inputs.push_back(expr_point.port.get_port_connector_ptr());
    for (const auto& expr_port : loop_exits)
        loop_end_inputs.push_back(expr_port.port.get_port_connector_ptr());
    loop_end_inputs.push_back(loop_begin_expr->get_output_port_connector(0));

    const auto& loop_end_expr = linear_ir.create_expression(loop_end, loop_end_inputs);
    linear_ir.insert(loop_end_pos, loop_end_expr);
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();

    std::set<size_t> inserted_loops;
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto expr = *expr_it;
        const auto& node = expr->get_node();
        if (ov::is_type<op::LoopBase>(node) ||
            ov::is_type<op::Buffer>(node) ||     // Need to cover Buffer
            ov::is_type<ov::op::v0::Parameter>(node) ||
            ov::is_type<ov::op::v0::Result>(node))
            continue;

        // Outer Loop ----> Inner Loop
        const auto expr_loops = expr->get_loop_ids();
        const auto loop_depth = expr_loops.size();
        for (size_t i = 0; i < loop_depth; ++i) {
            const auto loop_id = expr_loops[i];
            if (inserted_loops.count(loop_id) == 0) {
                const bool has_outer_loop = i > 0 && inserted_loops.find(expr_loops[i - 1]) != inserted_loops.end();
                insertion(linear_ir, loop_manager, loop_id, has_outer_loop);
                inserted_loops.insert(loop_id);  // save Loop ID
            }
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
