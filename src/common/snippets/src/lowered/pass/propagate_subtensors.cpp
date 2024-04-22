// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/propagate_subtensors.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
void propagate_updated_subtensor_through_loop(const LinearIR& linear_ir,
                                              const LinearIR::LoopManager::LoopInfoPtr& loop_info,
                                              LinearIR::container::const_iterator begin,
                                              LinearIR::container::const_iterator end,
                                              bool most_outer_loop,
                                              const size_t new_dim_value = SIZE_MAX) {
    OPENVINO_ASSERT(snippets::utils::implication(most_outer_loop, new_dim_value != SIZE_MAX),
                    "if the updated subtensor propagation was called for the outer loop, new_dim_value must not be equal to default value");
    std::map<lowered::PortDescriptorPtr, snippets::VectorDimsPtr> original_shapes;
    // First step: set new dim value to the corresponding entry_points' dimensions
    if (most_outer_loop) {
        for (const auto& port : loop_info->get_entry_points()) {
            const auto& reg_type = port.expr_port->get_descriptor_ptr()->get_reg().type;
            if ((port.is_incremented && reg_type == RegType::gpr) || (reg_type == RegType::vec)) {
                const auto& expr = port.expr_port->get_expr();
                const auto& desc = port.expr_port->get_descriptor_ptr();
                auto subtensor = desc->get_subtensor();
                if (port.dim_idx < subtensor.size()) {
                    *(subtensor.rbegin() + port.dim_idx) = new_dim_value;
                    desc->set_subtensor(subtensor);
                }

                const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();
                const auto& parent_shape = parent_desc->get_shape_ptr();
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = parent_shape;
                }
                auto new_shape = *parent_shape;
                new_shape[*(desc->get_layout().rbegin() + port.dim_idx)] = new_dim_value;
                parent_desc->set_shape_ptr(std::make_shared<VectorDims>(new_shape));
            }
        }
    }

    auto update_only_dim_idx_with_subtensor_value = [&](const LinearIR::LoopManager::LoopPort& port) {
        const auto& reg_type = port.expr_port->get_descriptor_ptr()->get_reg().type;
        if ((port.is_incremented && reg_type == RegType::gpr) || (reg_type == RegType::vec)) {
            const auto desc = port.expr_port->get_descriptor_ptr();
            const auto expr = port.expr_port->get_expr();
            const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();

            const auto& parent_shape = parent_desc->get_shape_ptr();
            const auto& desc_subtensor = desc->get_subtensor();
            if (port.dim_idx < desc_subtensor.size()) {
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = parent_shape;
                }
                auto new_shape = *parent_shape;
                new_shape[*(desc->get_layout().rbegin() + port.dim_idx)] = *(desc_subtensor.rbegin() + port.dim_idx);
                parent_desc->set_shape_ptr(std::make_shared<VectorDims>(new_shape));
            }
        }
    };

    auto update_subtensors = [](const std::vector<PortDescriptorPtr>& descs, bool is_input) {
        for (const auto& desc : descs) {
            const auto& subtensor = desc->get_subtensor();
            if (!subtensor.empty()) {
                auto planar_dims = is_input ? snippets::utils::get_planar_vdims(desc->get_shape(), desc->get_layout())
                                            : snippets::utils::get_preordered_vdims(desc->get_shape(), desc->get_layout());
                const size_t subtensor_start = planar_dims.size() - subtensor.size();
                VectorDims new_subtensor(planar_dims.begin() + subtensor_start, planar_dims.end());
                for (size_t i = 0; i < new_subtensor.size(); ++i) {
                    new_subtensor[i] = std::min(new_subtensor[i], subtensor[i]);
                }
                desc->set_subtensor(new_subtensor);
            }
        }
    };

    auto shape_inference_end_it = end;
    const bool loop_by_last_dim = loop_info->get_dim_idx() == 0;
    // Subtensors are updated using shape inference infrastructure:
    // For inner loops propagation function is called recursively
    for (auto expr_it = begin; expr_it != end; expr_it++) {
        const auto expr = *expr_it;
        if (ov::is_type<snippets::op::LoopEnd>(expr->get_node()))
            continue;
        if (auto loop_begin = ov::as_type_ptr<snippets::op::LoopBegin>(expr->get_node())) {
            const auto loop_end = loop_begin->get_loop_end();
            const auto inner_loop_info = linear_ir.get_loop_manager()->get_loop_info(loop_end->get_id());
            const auto inner_begin = std::next(expr_it);
            const auto inner_end = linear_ir.find_after(inner_begin, linear_ir.get_expr_by_node(loop_end));

            // The corresponding shapes of inner loops entry points must be updated using existing subtensor values
            if (!most_outer_loop) {
                for (const auto& port : loop_info->get_entry_points())
                    update_only_dim_idx_with_subtensor_value(port);
            }
            propagate_updated_subtensor_through_loop(linear_ir, inner_loop_info, inner_begin, inner_end, false);
            expr_it = inner_end;
            continue;
        }
        if ((ov::is_type<snippets::op::BroadcastMove>(expr_it->get()->get_node()) ||
            ov::is_type<snippets::op::BroadcastLoad>(expr_it->get()->get_node())) &&
            loop_by_last_dim) {
            // WA: we have to break subtensor propagation if we try to propagate new last dim through Broadcast nodes
            // which broadcast last dim in original dimension value anyway
            // This workaround might be avoided if blocked shape are used for tail size propagation
            shape_inference_end_it = expr_it;
            break;
        }
        expr->updateShapes();
        update_subtensors(expr->get_input_port_descriptors(), true);
        update_subtensors(expr->get_output_port_descriptors(), false);
    }

    // After subtensor propagation, the original shapes must be restored
    for (const auto& elem : original_shapes)
        elem.first->set_shape_ptr(elem.second);
    for (auto expr_it = begin; expr_it != shape_inference_end_it; expr_it++)
        (*expr_it)->updateShapes();
}
}  // namespace

UpdateSubtensors::UpdateSubtensors(size_t tail_size) : RangedPass(), m_tail_size(tail_size) {}

bool UpdateSubtensors::run(LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& last_expr = *end;
    const auto last_node = last_expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(last_node);
    OPENVINO_ASSERT(loop_end, "the last operation in range must be LoopEnd");

    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
    propagate_updated_subtensor_through_loop(linear_ir, loop_info, begin, end, true, m_tail_size);
    return true;
}

std::shared_ptr<pass::PassBase> UpdateSubtensors::merge(const std::shared_ptr<pass::PassBase>& other) {
    const auto merged_pass = std::make_shared<UpdateSubtensors>(m_tail_size);
    if (other == nullptr)
        return merged_pass;
    const auto casted_pass = ov::as_type_ptr<UpdateSubtensors>(other);
    if (!casted_pass || m_tail_size != casted_pass->m_tail_size)
        return nullptr;
    return merged_pass;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

