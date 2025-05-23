// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_broadcastmove.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InsertBroadcastMove::is_broadcasting_supported(const std::shared_ptr<ov::Node>& n) {
    return !std::dynamic_pointer_cast<modifier::MemoryAccess>(n) &&
           (ov::op::util::supports_auto_broadcast(n) ||
           n->get_autob().m_type == ov::op::AutoBroadcastType::NUMPY ||
           is_type<ov::op::v0::PRelu>(n));
}

bool InsertBroadcastMove::is_broadcasting_needed(const std::shared_ptr<ov::Node>& n) {
    // We don't need to insert BroadcastMove after the following operations:
    // - Scalar has emitter with explicit broadcasting
    // - VectorBuffer has scalar output shape to avoid broadcast conflicts and manually shape insertion.
    // - Fill can be inserted only after VectorBuffer, and should be ignored as well.
    return !utils::is_scalar_constant(n) &&
           !ov::is_type_any_of<ov::snippets::op::VectorBuffer, ov::snippets::op::Fill>(n);
}

std::vector<size_t> InsertBroadcastMove::get_last_dims(const ExpressionPtr& expr) {
    const auto& descriptors = expr->get_input_port_descriptors();
    std::vector<size_t> last_dims(descriptors.size());
    std::transform(descriptors.begin(), descriptors.end(), last_dims.begin(),
                   [](const std::shared_ptr<PortDescriptor>& d){
                        return d->get_shape().size() > 0 ? d->get_shape().back() : 1; });
    return last_dims;
}

size_t InsertBroadcastMove::get_max_dim(const std::vector<size_t>& last_dims) {
    // Specially set 0 to distinguish it from scalar or dynamic value (it's max value)
    size_t broadcast_dim = 0;
    for (const auto& last_dim : last_dims) {
        if (!utils::is_dynamic_value(last_dim) && last_dim > broadcast_dim)
            broadcast_dim = last_dim;
    }
    return broadcast_dim;
}

bool InsertBroadcastMove::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBroadcastMove")
    bool modified = false;

    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        if (!is_broadcasting_supported(node) || expr->get_input_count() < 2)
            continue;

        const auto last_dims = get_last_dims(expr);
        const auto broadcasted_dim = get_max_dim(last_dims);
        if (broadcasted_dim == 0)
            continue;

        for (size_t i = 0; i < last_dims.size(); i++) {
            const auto& input = expr->get_input_port_connector(i);
            const auto& parent_port = input->get_source();

            if (!utils::is_dynamic_value(last_dims[i]) && last_dims[i] != broadcasted_dim && is_broadcasting_needed(parent_port.get_expr()->get_node())) {
                OPENVINO_ASSERT(last_dims[i] == 1,
                                "Attempt to broadcast non-1 dimension. Target dim: ", broadcasted_dim, " This dim: ", last_dims[i]);
                const auto broadcast = std::make_shared<op::BroadcastMove>(node->get_input_source_output(i), broadcasted_dim);
                const auto broadcast_expr = *linear_ir.insert_node(broadcast, std::vector<PortConnectorPtr>{ input },
                                                                   expr->get_loop_ids(), true, expr_it, { expr->get_input_port(i) });
                // Note: We have to set live regs manually, since this transformation is applied after all register-related passes.
                // Since BroadcastMove sets in_regs the same as out_regs, live regs are the same as for the child.
                broadcast_expr->set_live_regs(expr->get_live_regs());
                // Note that BroadcastMove modified the next expr input shape, so we need to set update
                // expr's input port descriptor to reflect the changes
                expr->get_input_port_descriptor(i)->set_shape(broadcast_expr->get_output_port_descriptor(0)->get_shape());

                modified = true;
            }
        }
    }
    return modified;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

