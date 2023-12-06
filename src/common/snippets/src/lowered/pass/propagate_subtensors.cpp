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
                                              const size_t new_dim_value) {
    std::map<lowered::PortDescriptorPtr, snippets::VectorDims> original_shapes;
    static constexpr size_t existing_subtensor_value = SIZE_MAX;
    // First step: set new dim value to the corresponding entry_points' dimensions
    if (new_dim_value != existing_subtensor_value) {
        for (const auto& port : loop_info->get_entry_points()) {
            if (port.is_incremented) {
                const auto& expr = port.expr_port->get_expr();
                const auto node = expr->get_node();
                auto desc = port.expr_port->get_descriptor_ptr();
                auto subtensor = desc->get_subtensor();
                if (port.dim_idx < subtensor.size()) {
                    *(subtensor.rbegin() + port.dim_idx) = new_dim_value;
                    desc->set_subtensor(subtensor);
                }

                const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();
                const auto& layout = parent_desc->get_layout();
                const auto& shape = parent_desc->get_shape();
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = shape;
                }
                auto new_shape = shape;
                new_shape[*(layout.rbegin() + port.dim_idx)] = new_dim_value;
                parent_desc->set_shape(new_shape);
            }
        }
    }

    auto update_only_dim_idx_with_subtensor_value = [&](const LinearIR::LoopManager::LoopPort& port) {
        if (port.is_incremented) {
            auto desc = port.expr_port->get_descriptor_ptr();
            const auto expr = port.expr_port->get_expr();
            const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();

            const auto& layout = parent_desc->get_layout();
            const auto& shape = parent_desc->get_shape();
            const auto& desc_subtensor = desc->get_subtensor();
            if (port.dim_idx < desc_subtensor.size()) {
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = shape;
                }
                auto new_shape = shape;
                new_shape[*(layout.rbegin() + port.dim_idx)] = *(desc_subtensor.rbegin() + port.dim_idx);
                parent_desc->set_shape(new_shape);
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
            const auto inner_end = linear_ir.find(linear_ir.get_expr_by_node(loop_end));

            // The corresponding shapes of inner loops entry points must be updated using existing subtensor values
            if (new_dim_value == existing_subtensor_value) {
                for (const auto& port : loop_info->get_entry_points())
                    update_only_dim_idx_with_subtensor_value(port);
            }
            propagate_updated_subtensor_through_loop(linear_ir, inner_loop_info, inner_begin, inner_end, existing_subtensor_value);
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
        elem.first->set_shape(elem.second);
    for (auto expr_it = begin; expr_it != shape_inference_end_it; expr_it++)
        (*expr_it)->updateShapes();
}
}  // namespace

UpdateSubtensors::UpdateSubtensors(size_t tail_size) : SubgraphPass(), m_tail_size(tail_size) {}

bool UpdateSubtensors::run(const LinearIR& linear_ir, LinearIR::constExprIt begin, LinearIR::constExprIt end) {
    const auto& expr = *end;
    const auto node = expr->get_node();
    const auto loop_end = ov::as_type_ptr<op::LoopEnd>(node);
    const auto& loop_manager = linear_ir.get_loop_manager();
    const auto& loop_info = loop_manager->get_loop_info(loop_end->get_id());
    propagate_updated_subtensor_through_loop(linear_ir, loop_info, std::next(begin), end, m_tail_size);
    return true;
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

