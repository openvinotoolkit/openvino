// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "snippets/lowered/expressions/buffer_expression.hpp"
#include "snippets/op/loop.hpp"

#include "emitters/utils.hpp"

namespace ov {
namespace intel_cpu {
namespace utils {

size_t get_buffer_cluster_id(const ov::snippets::lowered::ExpressionPort& port, size_t offset) {
    auto get_cluster_id = [](const snippets::lowered::ExpressionPort& p) {
        const auto buffer = ov::as_type_ptr<ov::snippets::lowered::BufferExpression>(p.get_expr());
        return buffer ? buffer->get_cluster_id() : SIZE_MAX;
    };
    size_t id = SIZE_MAX;
    if (snippets::utils::is_dynamic_value(offset)) {
        switch (port.get_type()) {
            case (ov::snippets::lowered::ExpressionPort::Type::Input):
                id = get_cluster_id(port.get_port_connector_ptr()->get_source());
                break;
            case (ov::snippets::lowered::ExpressionPort::Type::Output):
                for (const auto& child : port.get_connected_ports())
                    if (!ov::is_type<snippets::op::LoopEnd>(child.get_expr()->get_node()))
                        id = get_cluster_id(child);
                break;
            default:
                OV_CPU_JIT_EMITTER_THROW("Uknown type of expression port!");
        }
        OV_CPU_JIT_EMITTER_ASSERT(id != SIZE_MAX, "Dynamic offset requires a valid buffer ID");
    }
    return id;
}

}   // namespace utils
}   // namespace intel_cpu
}   // namespace ov