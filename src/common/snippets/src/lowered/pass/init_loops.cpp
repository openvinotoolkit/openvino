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
int64_t get_input_stride(size_t dim, const std::vector<size_t>& layout, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (int i = static_cast<int>(layout.size()) - 1; i >= 0; i--) {
        if (layout[i] == dim) {
            break;
        }
        stride *= static_cast<int64_t>(shape[layout[i]]);
    }
    return stride;
}
int64_t get_output_stride(size_t dim, const std::vector<size_t>& shape) {
    int64_t stride = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        stride *= static_cast<int64_t>(shape[i]);
    }
    return stride;
}
}  // namespace

InitLoops::InitLoops() : Pass() {}

void InitLoops::init_ptr_increments(std::vector<LoopPort>& loop_inputs, std::vector<LoopPort>& loop_outputs, size_t work_amount, size_t dim_idx) {
    for (auto& loop_input : loop_inputs) {
        loop_input.ptr_increment = 0;
        if (loop_input.is_incremented) {
            const auto& port = loop_input.expr_port;
            const auto source = *port->get_connected_ports().begin();
            const auto loop_ids = port->get_expr()->get_loop_ids();
            const auto& layout = port->get_descriptor_ptr()->get_layout();
            const auto& shape = port->get_descriptor_ptr()->get_shape();
            const auto& dim = *(layout.rbegin() + dim_idx);
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(shape[dim] == 1 && work_amount != 1)) {
                // Input layout shows how we should read data by which order and strides
                loop_input.ptr_increment = get_input_stride(dim, source.get_descriptor_ptr()->get_layout(), shape);
            }
        }
    }

    for (auto& loop_output : loop_outputs) {
        loop_output.ptr_increment = 0;
        if (loop_output.is_incremented) {
            const auto& port = loop_output.expr_port;
            const auto loop_ids = port->get_expr()->get_loop_ids();
            const auto& layout = port->get_descriptor_ptr()->get_layout();
            const auto& shape = port->get_descriptor_ptr()->get_shape();
            const auto original_dim = layout.size() - 1 - dim_idx;
            const auto& dim = std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), original_dim));
            // If relevant dim is not broadcasted, then ptr_increment is the dim stride in the new layout
            if (!(shape[dim] == 1 && work_amount != 1)) {
                // Output layout shows how we already written data by which order and strides
                loop_output.ptr_increment = get_output_stride(dim, shape);
            }
        }
    }
}

void InitLoops::init_finalization_offsets(std::vector<LinearIR::LoopManager::LoopPort>& loop_inputs,
                                          std::vector<LinearIR::LoopManager::LoopPort>& loop_outputs,
                                          size_t work_amount) {
    for (auto& loop_input : loop_inputs) {
        loop_input.finalization_offset = -1 * loop_input.ptr_increment * work_amount;
    }
    for (auto& loop_output : loop_outputs) {
        loop_output.finalization_offset = -1 * loop_output.ptr_increment * work_amount;
    }
}

void InitLoops::init_element_type_sizes(std::vector<LoopPort>& loop_inputs,
                                        std::vector<LoopPort>& loop_outputs) {
    for (auto& loop_input : loop_inputs) {
        const auto& port = loop_input.expr_port;
        loop_input.data_size = static_cast<int64_t>(port->get_expr()->get_node()->get_input_element_type(port->get_index()).size());
    }
    for (auto& loop_output : loop_outputs) {
        const auto& port = loop_output.expr_port;
        loop_output.data_size = static_cast<int64_t>(port->get_expr()->get_node()->get_output_element_type(port->get_index()).size());
    }
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();
    for (const auto& loop : loops) {
        const auto loop_info = loop.second;

        const auto work_amount = loop_info->work_amount;
        const auto dim_idx = loop_info->dim_idx;

        init_ptr_increments(loop_info->entry_points, loop_info->exit_points, work_amount, dim_idx);
        init_finalization_offsets(loop_info->entry_points, loop_info->exit_points, work_amount);
        init_element_type_sizes(loop_info->entry_points, loop_info->exit_points);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
