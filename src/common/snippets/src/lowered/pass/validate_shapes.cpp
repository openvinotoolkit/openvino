// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/validate_shapes.hpp"

#include <cstddef>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/op/loop.hpp"

namespace ov::snippets::lowered::pass {

bool ValidateShapes::run([[maybe_unused]] LinearIR& linear_ir,
                         lowered::LinearIR::constExprIt begin,
                         lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ValidateShapes")

    for (auto expr_it = begin; expr_it != end; ++expr_it) {
        const auto& expr = *expr_it;
        const auto num_inputs = expr->get_input_count();
        const auto& port_connectors = expr->get_input_port_connectors();
        const auto& port_descriptors = expr->get_input_port_descriptors();
        OPENVINO_ASSERT(port_connectors.size() == num_inputs, "Invalid number of port connectors detected");
        OPENVINO_ASSERT(port_descriptors.size() == num_inputs, "Invalid number of port descriptors detected");
        for (size_t i = 0; i < num_inputs; i++) {
            if (ov::is_type<ov::snippets::op::LoopBase>(expr->get_node())) {
                continue;
            }
            const auto& descr = port_descriptors[i];
            const auto& layout = descr->get_layout();
            const auto& shape = descr->get_shape();
            const auto& n = expr->get_node();
            OPENVINO_ASSERT(layout.size() == shape.size(),
                            "Layout and shape sizes must match. ",
                            "Check the expr for node ",
                            n->get_friendly_name());
            const auto& parent_desc = port_connectors[i]->get_source().get_descriptor_ptr();
            const auto& parent_shape = parent_desc->get_shape();
            OPENVINO_ASSERT(parent_shape == shape,
                            "Parent shape must be equal to the expression shape. ",
                            "Check the expr for node ",
                            n->get_friendly_name());
        }
    }
    return false;
}

}  // namespace ov::snippets::lowered::pass
