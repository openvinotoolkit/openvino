// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/propagate_subtensors.hpp"

#include "snippets/lowered/linear_ir.hpp"
#include "snippets/lowered/loop_manager.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/itt.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {

// The algorithm uses the following special values in subtensors/shapes:
// 1. Dynamic value in subtensor/shape : SIZE_MAX
// 2. Full dimension in subtensor      : SIZE_MAX - 1
// 3. Default value of `new_dim_value` : SIZE_MAX - 2
// 4. `Forced` special dynamic value   : SIZE_MAX - 3
//
// We have to introduce `FORCED_DYNAMIC_VALUE` to distinguish `new_dim_value = DYNAMIC`
// from the real dynamic values in subtensors and shapes and force this value in subtensors.
// For example, there is Brgemm with the following info in the tail Loop:
// Input 0: shape [?, ?], existing subtensor [32, FULL_DIM]
// Input 1: shape [?, ?], existing subtensor [FULL_DIM, FULL_DIM]
// Output : shape [?, ?], existing subtensor [32, FULL_DIM]
// If the user wants to force `?` in the place of `32` in subtensors, the steps will be:
// 1. Set `?` to subtensor and shape of Input 0 :
//    shape [?, ?] (shape has not been changed!), new subtensor [?, FULL_DIM]
// 2. Make shape inference of Brgemm and get Output:
//    shape [?, ?] (shape has not been changed!), existing subtensor [FULL_DIM, FULL_DIM]
// 3. Update subtensor on output using shape:
//    new_subtensor[i] = std::min(planar_shape[i], subtensor[i]); // i = 0: std::min(SIZE_MAX(?), 32)
//    new subtensor [32, FULL_DIM] - has not been changed! But should be [?, FULL_DIM]
// Conclusion: we have to distinguish forced dynamic value with existing dynamic values in shape and subtensor

constexpr size_t NEW_DEFAULT_VALUE    = SIZE_MAX - 2;
constexpr size_t FORCED_DYNAMIC_VALUE = SIZE_MAX - 3;

void propagate_updated_subtensor_through_loop(const LinearIR& linear_ir,
                                              const LoopInfoPtr& loop_info,
                                              LinearIR::container::const_iterator begin,
                                              LinearIR::container::const_iterator end,
                                              bool most_outer_loop,
                                              size_t new_dim_value = NEW_DEFAULT_VALUE) {
    // Marks the forced dynamic value
    new_dim_value = utils::is_dynamic_value(new_dim_value) ? FORCED_DYNAMIC_VALUE : new_dim_value;
    OPENVINO_ASSERT(snippets::utils::implication(most_outer_loop, new_dim_value != NEW_DEFAULT_VALUE),
                    "if the updated subtensor propagation was called for the outer loop, new_dim_value must not be equal to default value");

    std::map<lowered::PortDescriptorPtr, snippets::VectorDims> original_shapes;
    // First step: set new dim value to the corresponding input_ports' dimensions
    if (most_outer_loop) {
        for (const auto& port : loop_info->get_input_ports()) {
            const auto& reg_type = port.expr_port->get_descriptor_ptr()->get_reg().type;
            if ((port.is_incremented && reg_type == RegType::gpr) || (reg_type == RegType::vec)) {
                const auto& expr = port.expr_port->get_expr();
                const auto& desc = port.expr_port->get_descriptor_ptr();
                auto subtensor = desc->get_subtensor();
                if (port.dim_idx < desc->get_subtensor().size()) {
                    desc->set_subtensor_dim(port.dim_idx, new_dim_value);
                }

                const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();
                const auto& parent_shape = parent_desc->get_shape();
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = parent_shape;
                }
                auto new_shape = parent_shape;
                new_shape[*(desc->get_layout().rbegin() + port.dim_idx)] = new_dim_value;
                parent_desc->set_shape(new_shape);
            }
        }
    }

    auto update_only_dim_idx_with_subtensor_value = [&](const LoopPort& port) {
        const auto& reg_type = port.expr_port->get_descriptor_ptr()->get_reg().type;
        if ((port.is_incremented && reg_type == RegType::gpr) || (reg_type == RegType::vec)) {
            const auto desc = port.expr_port->get_descriptor_ptr();
            const auto expr = port.expr_port->get_expr();
            const auto parent_desc = expr->get_input_port_connector(port.expr_port->get_index())->get_source().get_descriptor_ptr();

            const auto& parent_shape = parent_desc->get_shape();
            const auto& desc_subtensor = desc->get_subtensor();
            if (port.dim_idx < desc_subtensor.size()) {
                if (original_shapes.find(parent_desc) == original_shapes.end()) {
                    original_shapes[parent_desc] = parent_shape;
                }
                auto new_shape = parent_shape;
                new_shape[*(desc->get_layout().rbegin() + port.dim_idx)] = *(desc_subtensor.rbegin() + port.dim_idx);
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
                    // If user forces dynamic value to set in subtensor, set real dynamic dimension using `get_dynamic_value<size_t>()`
                    new_subtensor[i] = new_subtensor[i] == FORCED_DYNAMIC_VALUE ? utils::get_dynamic_value<size_t>() :
                                       utils::is_full_dim_value(subtensor[i]) ? subtensor[i] : std::min(new_subtensor[i], subtensor[i]);
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

            // The corresponding shapes of inner loops input ports must be updated using existing subtensor values
            if (!most_outer_loop) {
                for (const auto& port : loop_info->get_input_ports())
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
        elem.first->set_shape(elem.second);
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
    if (!other)
        return shared_from_this();
    const auto casted_pass = ov::as_type_ptr<UpdateSubtensors>(other);
    size_t merged_size;
    if (!casted_pass || !ov::snippets::utils::merge_dynamic_dim(merged_size, m_tail_size, casted_pass->m_tail_size))
        return nullptr;
    return std::make_shared<UpdateSubtensors>(merged_size);
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

