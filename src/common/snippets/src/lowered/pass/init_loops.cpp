// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

using LoopPort = LinearIR::LoopManager::LoopPort;

namespace {
inline int64_t get_stride(size_t dim, const VectorDims& shape) {
    int64_t stride = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        if (utils::is_dynamic_value(shape[i])) {
            return utils::get_dynamic_value<int64_t>();
        }
        stride *= static_cast<int64_t>(shape[i]);
    }
    return stride;
}

inline void init_is_incremented(LoopPort& loop_port) {
    if (!ov::is_type<op::MemoryAccess>(loop_port.expr_port->get_expr()->get_node()))
        loop_port.is_incremented = false;
}

inline void init_ptr_increment(LoopPort& loop_port, size_t work_amount) {
    loop_port.ptr_increment = 0;
    if (!loop_port.is_incremented)
        return;

    const auto& expr_port = loop_port.expr_port;
    const auto& layout = expr_port->get_descriptor_ptr()->get_layout();
    const auto& shape = expr_port->get_descriptor_ptr()->get_shape();
    size_t dim = 0;
    if (expr_port->get_type() == ExpressionPort::Input) {
        dim = utils::get_input_dim_idx(layout, loop_port.dim_idx);
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        dim = utils::get_output_dim_idx(layout, loop_port.dim_idx);
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
    // When we cannot say about broadcasting by last dim
    if (dim == shape.size() - 1 && utils::is_dynamic_value(shape.back())) {
        loop_port.ptr_increment = utils::get_dynamic_value<int64_t>();
    } else if (!(shape[dim] == 1 && work_amount != 1)) {
        loop_port.ptr_increment = get_stride(dim, shape);
    }
}

inline void init_finalization_offset(LoopPort& loop_port, size_t work_amount) {
    loop_port.finalization_offset =
        utils::is_dynamic_value(work_amount) || utils::is_dynamic_value(loop_port.ptr_increment) ? utils::get_dynamic_value<int64_t>()
                                                                                                 : -1 * loop_port.ptr_increment * work_amount;
}

inline void init_data_size(LoopPort& loop_port) {
    const auto& expr_port = loop_port.expr_port;
    if (expr_port->get_type() == ExpressionPort::Input) {
        loop_port.data_size = static_cast<int64_t>(expr_port->get_expr()->get_node()->get_input_element_type(expr_port->get_index()).size());
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        loop_port.data_size = static_cast<int64_t>(expr_port->get_expr()->get_node()->get_output_element_type(expr_port->get_index()).size());
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
}

inline void init_work_amount(const LinearIR::LoopManager::LoopInfoPtr& loop_info) {
    size_t work_amount = 1;
    for (const auto& loop_port : loop_info->get_entry_points()) {
        if (loop_port.is_incremented) {
            const auto& desc = loop_port.expr_port->get_descriptor_ptr();
            const auto& shape = desc->get_shape();
            const auto& layout = desc->get_layout();
            utils::broadcast_merge_dim(work_amount, work_amount, shape[utils::get_input_dim_idx(layout, loop_port.dim_idx)]);
        }
    }
    for (const auto& loop_port : loop_info->get_exit_points()) {
        if (loop_port.is_incremented) {
            const auto& desc = loop_port.expr_port->get_descriptor_ptr();
            const auto& shape = desc->get_shape();
            const auto& layout = desc->get_layout();
            utils::broadcast_merge_dim(work_amount, work_amount, shape[utils::get_output_dim_idx(layout, loop_port.dim_idx)]);
        }
    }
    loop_info->set_work_amount(work_amount);
}
}  // namespace

void InitLoops::init_loop_info(const LinearIR::LoopManager::LoopInfoPtr& loop_info, bool only_runtime_args) {
    if (utils::is_dynamic_value(loop_info->get_work_amount()))
        init_work_amount(loop_info);

    const auto work_amount = loop_info->get_work_amount();

    auto init_runtime_parameters = [&work_amount](LoopPort& loop_port) {
        init_ptr_increment(loop_port, work_amount);
        init_finalization_offset(loop_port, work_amount);
    };

    auto init_all_parameters = [&init_runtime_parameters](LoopPort& loop_port) {
        init_is_incremented(loop_port);
        init_data_size(loop_port);
        init_runtime_parameters(loop_port);
    };

    if (only_runtime_args) {
        loop_info->update_entry_points(init_runtime_parameters);
        loop_info->update_exit_points(init_runtime_parameters);
    } else {
        loop_info->update_entry_points(init_all_parameters);
        loop_info->update_exit_points(init_all_parameters);
    }
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();
    for (const auto& loop : loops) {
        init_loop_info(loop.second);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
