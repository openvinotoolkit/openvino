// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "emitters/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
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

}  // namespace ov::intel_cpu::utils
