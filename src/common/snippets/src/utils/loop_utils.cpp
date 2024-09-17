// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils/loop_utils.hpp"

#include "snippets/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace utils {

using namespace ov::snippets::lowered;
namespace {
inline int64_t get_ptr_increment(const LoopPort& loop_port, size_t work_amount, size_t port_count) {
    if (!loop_port.is_incremented)
        return 0;

    const auto& expr_port = loop_port.expr_port;
    const auto& layout = expr_port->get_descriptor_ptr()->get_layout();
    const auto& shape = expr_port->get_descriptor_ptr()->get_shape();
    size_t dim = 0;
    if (expr_port->get_type() == ExpressionPort::Input) {
        dim = get_input_dim_idx(layout, loop_port.dim_idx);
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        dim = get_output_dim_idx(layout, loop_port.dim_idx);
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
    // When we cannot say about broadcasting
    if (is_dynamic_value(shape[dim]) && port_count > 1) {
        return get_dynamic_value<int64_t>();
    } else if (!(shape[dim] == 1 && work_amount != 1)) {
        return get_stride(dim, shape);
    }
    return 0;
}

inline int64_t get_finalization_offset(size_t work_amount, int64_t ptr_increment) {
    if (ptr_increment == 0 || work_amount == 0)
        return 0;
    if (is_dynamic_value(work_amount) || is_dynamic_value(ptr_increment))
        return get_dynamic_value<int64_t>();
    return -1 * ptr_increment * work_amount;
}

inline void init_work_amount(const LoopInfoPtr& loop_info) {
    size_t work_amount = 1;
    loop_info->iterate_through_ports([&work_amount](const LoopPort& loop_port) {
        if (loop_port.is_incremented) {
            const auto& desc = loop_port.expr_port->get_descriptor_ptr();
            const auto& shape = desc->get_shape();
            const auto& layout = desc->get_layout();
            const auto is_input = loop_port.expr_port->get_type() == ExpressionPort::Input;
            const auto dim_idx = is_input ? get_input_dim_idx(layout, loop_port.dim_idx) : get_output_dim_idx(layout, loop_port.dim_idx);
            OPENVINO_ASSERT(broadcast_merge_dim(work_amount, work_amount, shape[dim_idx]),
                            "Failed to broadcast work_amount");
        }
    });
    loop_info->set_work_amount(work_amount);
}
}  // namespace

void update_data_pointer_shifts(const UnifiedLoopInfoPtr& loop_info) {
    OPENVINO_ASSERT(loop_info != nullptr, "UnifiedLoopInfo is nullptr, nothing to update");
    const auto work_amount = loop_info->get_work_amount();
    const auto input_count = loop_info->get_input_count();
    const auto output_count = loop_info->get_output_count();

    auto update_shifts = [&work_amount, &input_count, &output_count](LoopPort& loop_port, UnifiedLoopInfo::LoopPortDesc& ptr_shifts_params) {
        ptr_shifts_params.ptr_increment = get_ptr_increment(loop_port, work_amount,
                                                            loop_port.expr_port->get_type() == ExpressionPort::Input ? input_count : output_count);
        ptr_shifts_params.finalization_offset = get_finalization_offset(work_amount, ptr_shifts_params.ptr_increment);
    };
    loop_info->iterate_through_infos(update_shifts);
}

void update_runtime_parameters(const UnifiedLoopInfoPtr& loop_info) {
    OPENVINO_ASSERT(loop_info != nullptr, "UnifiedLoopInfo is nullptr, nothing to update");
    if (!ov::is_type<InnerSplittedUnifiedLoopInfo>(loop_info))
        init_work_amount(loop_info);
    update_data_pointer_shifts(loop_info);
}

} // namespace utils
} // namespace snippets
} // namespace ov