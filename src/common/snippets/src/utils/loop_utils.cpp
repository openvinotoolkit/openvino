// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/utils/loop_utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/loop_info.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/lowered/loop_port.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::utils {

using namespace ov::snippets::lowered;
namespace {
inline int64_t get_ptr_increment(const LoopInfoPtr& outer_split_info_of_nested_loop,
                                 const LoopPort& loop_port,
                                 size_t work_amount,
                                 size_t port_count) {
    if (!loop_port.is_incremented()) {
        return 0;
    }
    const auto& layout = loop_port.get_expr_port()->get_descriptor_ptr()->get_layout();
    const auto port_type = loop_port.get_expr_port()->get_type();
    auto get_port_dim_idx = [&layout, &port_type](size_t dim_idx) {
        if (port_type == ExpressionPort::Input) {
            return get_input_dim_idx(layout, dim_idx);
        }
        if (port_type == ExpressionPort::Output) {
            return get_output_dim_idx(layout, dim_idx);
        }
        OPENVINO_THROW("Unsupported expression port type!");
    };

    const auto& expr_port = loop_port.get_expr_port();
    const auto& original_shape = expr_port->get_descriptor_ptr()->get_shape();
    const auto dim_idx = get_port_dim_idx(loop_port.get_dim_idx());
    // When we cannot say about broadcasting
    if (is_dynamic_value(original_shape[dim_idx]) && port_count > 1) {
        return get_dynamic_value<int64_t>();
    }
    if (original_shape[dim_idx] != 1 || work_amount == 1) {
        auto shape_for_stride_calculation = original_shape;
        // Note: in case of outer split loop, we may need to stride not the whole dimension, but only block size.
        // Example:
        // <other loops>
        // |  LoopBegin (outer_split): wa = n, inc = n_blk, dim_idx = 0
        // |  |  ...
        // |  |  Buffer_in [m_blk x n_blk]
        // |  |  LoopBegin (cur_loop): wa = m_blk, inc = 1, dim_idx = 1
        // |  |  |  Load
        // |  |  |  ...
        // |  |  |  Store
        // |  |  LoopEnd (cur_loop)
        // |  LoopEnd (outer_split)
        // |   Buffer_out [m_blk x n]
        // <other loops>
        // -------------
        // In this case, cur_loop ptr increments must be the following:
        // - Load from Buffer_in: ptr_increment = n_blk, since this loop port is inside outer_split loop
        // - Store to Buffer_out: ptr_increment = n, since this loop port is outside outer_split loop
        if (outer_split_info_of_nested_loop != nullptr) {
            const auto& ports = port_type == ExpressionPort::Input
                                    ? outer_split_info_of_nested_loop->get_input_ports()
                                    : outer_split_info_of_nested_loop->get_output_ports();
            auto it = std::find_if(ports.cbegin(), ports.cend(), [&expr_port](const LoopPort& lp) {
                return *lp.get_expr_port() == *expr_port;
            });
            if (it == ports.cend()) {
                const auto shape_dim_idx = get_port_dim_idx(outer_split_info_of_nested_loop->get_dim_idx());
                shape_for_stride_calculation[shape_dim_idx] = outer_split_info_of_nested_loop->get_increment();
            }
        }
        return get_stride(dim_idx, shape_for_stride_calculation);
    }
    return 0;
}

inline int64_t get_finalization_offset(size_t work_amount, int64_t ptr_increment) {
    if (any_of(0U, ptr_increment, work_amount)) {
        return 0;
    }
    if (is_dynamic_value(work_amount) || is_dynamic_value(ptr_increment)) {
        return get_dynamic_value<int64_t>();
    }
    return -1 * ptr_increment * work_amount;
}

inline void init_work_amount(const LoopInfoPtr& loop_info) {
    size_t work_amount = 1;
    loop_info->iterate_through_ports([&work_amount](const LoopPort& loop_port) {
        if (loop_port.is_processed()) {
            const auto& desc = loop_port.get_expr_port()->get_descriptor_ptr();
            const auto& shape = desc->get_shape();
            const auto& layout = desc->get_layout();
            const auto is_input = loop_port.get_expr_port()->get_type() == ExpressionPort::Input;
            const auto dim_idx = is_input ? get_input_dim_idx(layout, loop_port.get_dim_idx())
                                          : get_output_dim_idx(layout, loop_port.get_dim_idx());
            OPENVINO_ASSERT(broadcast_merge_dim(work_amount, work_amount, shape[dim_idx]),
                            "Failed to broadcast work_amount");
        }
    });
    loop_info->set_work_amount(work_amount);
}
}  // namespace

void update_data_pointer_shifts(const LoopManagerPtr& loop_manager, const UnifiedLoopInfoPtr& loop_info) {
    OPENVINO_ASSERT(loop_info != nullptr, "UnifiedLoopInfo is nullptr, nothing to update");
    const auto work_amount = loop_info->get_work_amount();
    const auto input_count = loop_info->get_input_count();
    const auto output_count = loop_info->get_output_count();

    // WA: to find outer split loop whose dim_idx is less than cur_dim_idx,
    // we use the knowledge that such outer loop is connected with the inner split loop
    // which is nested inside the current loop
    // TODO: this logic must be reworked, and WA should be removed, when blocking shapes are supported
    // Ticket: 155651
    LoopInfoPtr outer_split_info_of_nested_loop = nullptr;
    if (auto cur_dim_idx = loop_info->get_dim_idx(); cur_dim_idx != LoopPort::UNDEFINED_DIM_IDX) {
        auto fst_port_expr = loop_info->get_input_ports().front().get_expr_port()->get_expr();
        for (const auto loop_idx : fst_port_expr->get_loop_ids()) {
            const auto loop_info = loop_manager->get_loop_info(loop_idx);
            if (const auto inner_split_loop = ov::as_type_ptr<InnerSplittedUnifiedLoopInfo>(loop_info)) {
                if (inner_split_loop->get_dim_idx() < cur_dim_idx) {
                    OPENVINO_ASSERT(outer_split_info_of_nested_loop == nullptr,
                                    "only 1 nested inner split loop is supported");
                    outer_split_info_of_nested_loop = inner_split_loop->get_outer_splitted_loop_info();
                }
            }
        }
    }

    auto update_shifts = [&](LoopPort& loop_port, UnifiedLoopInfo::LoopPortDesc& ptr_shifts_params) {
        ptr_shifts_params.ptr_increment = get_ptr_increment(
            outer_split_info_of_nested_loop,
            loop_port,
            work_amount,
            loop_port.get_expr_port()->get_type() == ExpressionPort::Input ? input_count : output_count);
        ptr_shifts_params.finalization_offset = get_finalization_offset(work_amount, ptr_shifts_params.ptr_increment);
    };
    loop_info->iterate_through_infos(update_shifts);
}

void update_runtime_parameters(const LoopManagerPtr& loop_manager, const UnifiedLoopInfoPtr& loop_info) {
    OPENVINO_ASSERT(loop_info != nullptr, "UnifiedLoopInfo is nullptr, nothing to update");
    if (!ov::is_type<InnerSplittedUnifiedLoopInfo>(loop_info)) {
        init_work_amount(loop_info);
    }
    update_data_pointer_shifts(loop_manager, loop_info);
}

bool should_be_loop_port(const ov::snippets::lowered::ExpressionPort& port, size_t loop_id) {
    const auto& connected_ports = port.get_connected_ports();
    return std::any_of(connected_ports.cbegin(), connected_ports.cend(), [&](const ExpressionPort& connected_port) {
        const auto& loops = connected_port.get_expr()->get_loop_ids();
        return std::find(loops.cbegin(), loops.cend(), loop_id) == loops.cend();
    });
}

}  // namespace ov::snippets::utils
