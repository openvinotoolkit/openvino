// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/propagate_layout.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool PropagateLayout::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::PropagateLayout")
    if (linear_ir.empty())
        return false;

    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto io_expr = std::dynamic_pointer_cast<IOExpression>(expr);
        if (!io_expr)
            continue;

        const bool is_input = io_expr->get_type() == IOExpression::io_type::INPUT;
        const auto& connectors = is_input ? expr->get_output_port_connectors() : expr->get_input_port_connectors();
        if (connectors.size() != 1)
            OPENVINO_THROW("Parameter/Results should have exactly one output/input");

        // If input - we should be looking downstream, if output - upstream
        const auto& target_connector = connectors.front();
        if (is_input) {
            const auto consumer_inputs = target_connector->get_consumers();
            // Note that here we consider only the first child (which is usually load),
            // but often there is another child - LoopEnd
            std::set<std::vector<size_t>> child_layouts;
            for (const auto& child_input : consumer_inputs) {
                const auto& child = child_input.get_expr();
                const auto port = child_input.get_index();
                const auto& n = child->get_node();
                const auto ma = ov::as_type_ptr<op::MemoryAccess>(n);
                if (ma && ma->is_memory_access_input_port(port)) {
                    child_layouts.insert(child_input.get_descriptor_ptr()->get_layout());
                }
            }
            OPENVINO_ASSERT(child_layouts.size() == 1, "All children of an input expression must have the same layout");
            io_expr->get_output_port_descriptor(0)->set_layout(*child_layouts.begin());
        } else {
            io_expr->get_input_port_descriptor(0)->set_layout(target_connector->get_source().get_descriptor_ptr()->get_layout());
        }
    }

    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov
