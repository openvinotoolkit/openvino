// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"
#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/loop.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils/utils.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::utils {

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port) {
    auto get_cluster_id = [](const ov::snippets::lowered::ExpressionPort& p) {
        const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(p.get_expr());
        return buffer ? buffer->get_cluster_id() : SIZE_MAX;
    };
    const auto& ma_op = std::dynamic_pointer_cast<ov::snippets::modifier::MemoryAccess>(port.get_expr()->get_node());
    OPENVINO_ASSERT(ma_op, "Expected MemoryAccess op!");
    auto offset = ov::snippets::utils::get_dynamic_value<size_t>();
    size_t id = SIZE_MAX;
    switch (port.get_type()) {
    case ov::snippets::lowered::ExpressionPort::Type::Input:
        offset = ma_op->get_input_offset(port.get_index());
        id = get_cluster_id(port.get_port_connector_ptr()->get_source());
        break;
    case ov::snippets::lowered::ExpressionPort::Type::Output:
        offset = ma_op->get_output_offset(port.get_index());
        for (const auto& child : port.get_connected_ports()) {
            if (!ov::is_type<ov::snippets::op::LoopEnd>(child.get_expr()->get_node())) {
                id = get_cluster_id(child);
            }
        }
        break;
    default:
        OV_CPU_JIT_EMITTER_THROW("Uknown type of expression port!");
    }
    OV_CPU_JIT_EMITTER_ASSERT(implication(ov::snippets::utils::is_dynamic_value(offset), id != SIZE_MAX),
                              "In dynamic case Buffer Cluster ID must be known!");
    return id;
}

size_t get_parent_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr, "Expression must not be null");
    OPENVINO_ASSERT(expr->get_input_count() == 1, "MemoryAccess must have one parent");
    return get_buffer_cluster_id(expr->get_input_port(0));
}

size_t get_consumer_buffer_cluster_id(const ov::snippets::lowered::ExpressionPtr& expr) {
    OPENVINO_ASSERT(expr, "Expression must not be null");
    OPENVINO_ASSERT(expr->get_output_count() == 1, "MemoryAccess must have one output");
    return get_buffer_cluster_id(expr->get_output_port(0));
}

jit_snippets_call_args::loop_args_t compose_loop_args(const std::shared_ptr<ov::snippets::op::LoopEnd>& loop_end) {
    const auto& ptr_increments = loop_end->get_ptr_increments();
    const auto& fin_offsets = loop_end->get_finalization_offsets();
    const auto& is_incremented = loop_end->get_is_incremented();
    const auto wa_increment = loop_end->get_increment();

    const auto int_work_amount = ov::snippets::utils::is_dynamic_value(loop_end->get_work_amount())
                                     ? ov::snippets::utils::get_dynamic_value<int64_t>()
                                     : static_cast<int64_t>(loop_end->get_work_amount());
    auto loop_args = jit_snippets_call_args::loop_args_t(int_work_amount, ptr_increments, fin_offsets);

    const auto& data_sizes = loop_end->get_element_type_sizes();
    for (int64_t i = 0; i < loop_args.m_num_data_ptrs; ++i) {
        if (!is_incremented[i]) {
            loop_args.m_ptr_increments[i] = 0;
            loop_args.m_finalization_offsets[i] = 0;
            continue;
        }

        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_ptr_increments[i])) {
            loop_args.m_ptr_increments[i] *= (wa_increment * data_sizes[i]);
        }
        if (!ov::snippets::utils::is_dynamic_value(loop_args.m_finalization_offsets[i])) {
            loop_args.m_finalization_offsets[i] *= data_sizes[i];
        }
    }

    return loop_args;
}

}  // namespace ov::intel_cpu::utils
