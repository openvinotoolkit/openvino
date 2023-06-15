// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/insert_broadcastmove.hpp"
#include "snippets/utils.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

bool InsertBroadcastMove::run(LinearIR& linear_ir) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::InsertBroadcastMove")
    bool modified = false;

    auto supports_broadcasting = [](const std::shared_ptr<ov::Node>& n) {
      return ov::op::util::supports_auto_broadcast(n) ||
             n->get_autob().m_type == ov::op::AutoBroadcastType::NUMPY ||
             is_type<ov::op::v0::PRelu>(n);
    };
    auto dont_need_broadcasting = [](const ov::Output<ov::Node>& v){
        // We don't need to insert BroadcastMove after the following operations:
        // - Scalar has emitter with explicit broadcasting
        // - VectorBuffer has scalar output shape to avoid broadcast conflicts and manually shape insertion.
        // - Fill can be inserted only after VectorBuffer, and should be ignored as well.
        return utils::is_scalar_constant(v.get_node_shared_ptr()) ||
               ov::is_type<ov::snippets::op::VectorBuffer>(v.get_node_shared_ptr()) ||
                ov::is_type<ov::snippets::op::Fill>(v.get_node_shared_ptr());
    };
    for (auto expr_it = linear_ir.begin(); expr_it != linear_ir.end(); expr_it++) {
        const auto& expr = *expr_it;
        const auto& node = expr->get_node();
        const auto& descriptors = expr->get_input_port_descriptors();
        if (!supports_broadcasting(node) || descriptors.size() < 2)
            continue;
        const auto& connectors = expr->get_input_port_connectors();
        OPENVINO_ASSERT(connectors.size() == descriptors.size(),
                        "Invalid expression configuration: connectors and descriptors size mismatch");

        std::vector<size_t> last_dims(descriptors.size());
        std::transform(descriptors.begin(), descriptors.end(), last_dims.begin(),
                       [](const std::shared_ptr<PortDescriptor>& d){
                           return d->get_shape().back();
                       });
        const auto broadcasted_dim = *std::max_element(last_dims.begin(), last_dims.end());
        for (size_t i = 0; i < last_dims.size(); i++) {
            const auto& parent_port = connectors[i]->get_source();
            if (last_dims[i] != broadcasted_dim &&
                !dont_need_broadcasting(parent_port.get_expr()->get_node())) {
                OPENVINO_ASSERT(last_dims[i] == 1,
                                "Attempt to broadcast non-1 dimension. Target dim: ", broadcasted_dim,
                                " This dim: ", last_dims[i]);
                auto input_shape = descriptors[i]->get_shape();
                // Note that input_shape could be empty (aka ngraph scalar), so we can't just replace the last dim
                if (input_shape.empty())
                    input_shape.resize(1);
                input_shape.back() = last_dims[i];
                const auto broadcast = std::make_shared<op::BroadcastMove>(node->get_input_source_output(i), utils::vdims_to_pshape(input_shape));

                PortDescriptorUtils::set_port_descriptor_ptr(broadcast->output(0), connectors[i]->get_source().get_descriptor_ptr()->clone());
                const auto broadcast_expr = linear_ir.create_expression(broadcast, {connectors[i]});
                linear_ir.insert(expr_it, broadcast_expr);
                linear_ir.replace_input(expr->get_input_port(i), broadcast_expr->get_output_port_connector(0));
                // Note that BroadcastMove modified the next expr input shape, so we need to set update
                // expr's input port descriptor to reflect the changes
                expr->get_input_port_descriptor(i)->set_shape(broadcast_expr->get_output_port_descriptor(0)->get_shape());

                // Copy Loop identifies
                broadcast_expr->set_loop_ids(expr->get_loop_ids());

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

