// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using LoopPort = LinearIR::LoopManager::LoopPort;

namespace {
int64_t get_input_stride(size_t dim, const std::vector<size_t>& layout, const VectorDims& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim) {
            break;
        }
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
int64_t get_output_stride(size_t dim, const VectorDims& shape) {
    int64_t stride = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        stride *= static_cast<int64_t>(shape[i]);
    }
    return stride;
}
}  // namespace

InitLoops::InitLoops() : Pass() {}

void InitLoops::init_ptr_increments(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    const auto work_amount = loop_info->get_work_amount();
    auto loop_entries = loop_info->get_entry_points();
    auto loop_exits = loop_info->get_exit_points();

    for (auto& loop_entry : loop_entries) {
        loop_entry.ptr_increment = 0;
        if (loop_entry.is_incremented) {
            const auto& port = loop_entry.expr_port;
            const auto source = *port->get_connected_ports().begin();
            const auto& layout = port->get_descriptor_ptr()->get_layout();
            const auto& shape = port->get_descriptor_ptr()->get_shape();
            const auto& dim = *(layout.rbegin() + loop_entry.dim_idx);
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(shape[dim] == 1 && work_amount != 1)) {
                // Input layout shows how we should read data by which order and strides
                loop_entry.ptr_increment = get_input_stride(dim, source.get_descriptor_ptr()->get_layout(), shape);
            }
        }
    }

    for (auto& loop_exit : loop_exits) {
        loop_exit.ptr_increment = 0;
        if (loop_exit.is_incremented) {
            const auto& port = loop_exit.expr_port;
            const auto& layout = port->get_descriptor_ptr()->get_layout();
            const auto& shape = port->get_descriptor_ptr()->get_shape();
            const auto original_dim = layout.size() - 1 - loop_exit.dim_idx;
            const auto& dim = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), original_dim));
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(shape[dim] == 1 && work_amount != 1)) {
                // Output layout shows how we already written data by which order and strides
                loop_exit.ptr_increment = get_output_stride(dim, shape);
            }
        }
    }
    loop_info->set_entry_points(loop_entries);
    loop_info->set_exit_points(loop_exits);
}

void InitLoops::init_finalization_offsets(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    const auto work_amount = loop_info->get_work_amount();
    auto loop_entries = loop_info->get_entry_points();
    auto loop_exits = loop_info->get_exit_points();
    for (auto& loop_entry : loop_entries) {
        loop_entry.finalization_offset = -1 * loop_entry.ptr_increment * work_amount;
    }
    for (auto& loop_exit : loop_exits) {
        loop_exit.finalization_offset = -1 * loop_exit.ptr_increment * work_amount;
    }
    loop_info->set_entry_points(loop_entries);
    loop_info->set_exit_points(loop_exits);
}

void InitLoops::init_element_type_sizes(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    auto loop_entries = loop_info->get_entry_points();
    auto loop_exits = loop_info->get_exit_points();
    for (auto& loop_entry : loop_entries) {
        const auto& port = loop_entry.expr_port;
        loop_entry.data_size = static_cast<int64_t>(port->get_expr()->get_node()->get_input_element_type(port->get_index()).size());
    }
    for (auto& loop_exit : loop_exits) {
        const auto& port = loop_exit.expr_port;
        loop_exit.data_size = static_cast<int64_t>(port->get_expr()->get_node()->get_output_element_type(port->get_index()).size());
    }
    loop_info->set_entry_points(loop_entries);
    loop_info->set_exit_points(loop_exits);
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();
    for (const auto& loop : loops) {
        const auto loop_info = loop.second;
        init_ptr_increments(loop_info);
        init_finalization_offsets(loop_info);
        init_element_type_sizes(loop_info);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
