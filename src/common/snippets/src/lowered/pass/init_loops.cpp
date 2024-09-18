// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/init_loops.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/buffer.hpp"
#include "snippets/op/memory_access.hpp"
#include "snippets/utils/loop_utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

namespace {
inline void init_is_incremented(LoopPort& port, size_t loop_id) {
    const auto& expr = port.expr_port->get_expr();
    const auto& expr_loops = expr->get_loop_ids();
    if (!std::dynamic_pointer_cast<modifier::MemoryAccess>(expr->get_node())) {
        port.is_incremented = false;
    } else if (expr_loops.back() != loop_id) {
        // Note: LoopPort connected to Buffer between two loops should not be incremented in the outermost loop
        // Consider the example below:
        //     Store; Loop ids [0,1,2,3]
        //     Buffer; Loop ids [0,1]
        //     Load; Loop ids [0,1,4,5]
        // Store is output port of Loop-1, but it should be incremented only in Loop-2 and Loop-3. Similar with Load.
        auto is_ignored = [=](const ExpressionPtr& target_expr) {
            if (ov::is_type<BufferExpression>(target_expr)) {
                const auto& target_loops = target_expr->get_loop_ids();
                const auto i_max = std::min(expr_loops.size(), target_loops.size());
                for (size_t i = 0; i < i_max && expr_loops[i] == target_loops[i]; i++) {
                    if (target_loops[i] == loop_id)
                        return true;
                }
            }
            return false;
        };
        if (port.expr_port->get_type() == ExpressionPort::Type::Output) {
            const auto& out_connector = expr->get_output_port_connector(port.expr_port->get_index());
            for (const auto& consumer : out_connector->get_consumers()) {
                if (is_ignored(consumer.get_expr())) {
                    port.is_incremented = false;
                    return;
                }
            }
        } else if (port.expr_port->get_type() == ExpressionPort::Type::Input) {
            const auto& in_connector = expr->get_input_port_connector(port.expr_port->get_index());
            if (is_ignored(in_connector->get_source().get_expr())) {
                port.is_incremented = false;
                return;
            }
        } else {
            OPENVINO_THROW("Unexpected LoopPort type");
        }
    }
}

inline int64_t get_data_size(const LoopPort& loop_port) {
    const auto& expr_port = loop_port.expr_port;
    if (expr_port->get_type() == ExpressionPort::Input) {
        return static_cast<int64_t>(expr_port->get_expr()->get_node()->get_input_element_type(expr_port->get_index()).size());
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        return static_cast<int64_t>(expr_port->get_expr()->get_node()->get_output_element_type(expr_port->get_index()).size());
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
}
}  // namespace

void InitLoops::update_compile_parameters(const UnifiedLoopInfoPtr& loop_info, size_t loop_id) {
    OPENVINO_ASSERT(loop_info != nullptr, "UnifiedLoopInfo is nullptr, nothing to update");
    loop_info->iterate_through_infos(
        [loop_id](LoopPort& loop_port, UnifiedLoopInfo::LoopPortDesc& ptr_shifts_params) {
            init_is_incremented(loop_port, loop_id);
            ptr_shifts_params.data_size = get_data_size(loop_port);
        });
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loops = loop_manager->get_map();
    for (const auto& loop : loops) {
        const auto& loop_id = loop.first;
        const auto& loop_info = ov::as_type_ptr<UnifiedLoopInfo>(loop.second);
        update_compile_parameters(loop_info, loop_id);
        ov::snippets::utils::update_runtime_parameters(loop_info);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
