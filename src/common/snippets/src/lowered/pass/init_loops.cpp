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
inline void init_is_incremented(LoopPort& port) {
    const auto& expr = port.get_expr_port()->get_expr();
    if (!std::dynamic_pointer_cast<modifier::MemoryAccess>(expr->get_node())) {
        port.convert_to_type<LoopPort::Type::NotIncremented>();
    }
}

inline int64_t get_data_size(const LoopPort& loop_port) {
    const auto& expr_port = loop_port.get_expr_port();
    if (expr_port->get_type() == ExpressionPort::Input) {
        return static_cast<int64_t>(expr_port->get_expr()->get_node()->get_input_element_type(expr_port->get_index()).size());
    } else if (expr_port->get_type() == ExpressionPort::Output) {
        return static_cast<int64_t>(expr_port->get_expr()->get_node()->get_output_element_type(expr_port->get_index()).size());
    } else {
        OPENVINO_THROW("Unsupported expression port type!");
    }
}
}  // namespace

void InitLoops::update_compile_parameters(const UnifiedLoopInfoPtr& loop_info) {
    OPENVINO_ASSERT(loop_info != nullptr, "UnifiedLoopInfo is nullptr, nothing to update");
    loop_info->iterate_through_infos(
        [](LoopPort& loop_port, UnifiedLoopInfo::LoopPortDesc& ptr_shifts_params) {
            init_is_incremented(loop_port);
            ptr_shifts_params.data_size = get_data_size(loop_port);
        });
}

bool InitLoops::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InitLoops")
    if (linear_ir.empty())
        return false;

    const auto& loops = linear_ir.get_loop_manager()->get_map();
    for (const auto& loop : loops) {
        const auto& loop_info = ov::as_type_ptr<UnifiedLoopInfo>(loop.second);
        update_compile_parameters(loop_info);
        ov::snippets::utils::update_runtime_parameters(loop_info);
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
